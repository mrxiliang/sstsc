# -*- coding: utf-8 -*-

import torch
from optim.pytorchtools import EarlyStopping
import torch.nn as nn
from utils import misc
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import contextlib
import numpy as np
from utils.loss import entropy_y_x
from utils.loss import mse_with_softmax
from utils.loss import kl_div_with_logit
from utils.ramps import exp_rampup
from utils.context import disable_tracking_bn_stats


class RelationalReasoning_SupPF(torch.nn.Module):

  def __init__(self, backbone, feature_size=64, nb_class=3):
    super(RelationalReasoning_SupPF, self).__init__()
    self.backbone = backbone
    self.relation_head = torch.nn.Sequential(
                             torch.nn.Linear(feature_size*3, 256),
                             torch.nn.BatchNorm1d(256),
                             torch.nn.LeakyReLU(),
                             torch.nn.Linear(256, 1))
    self.sup_head = torch.nn.Sequential(
        torch.nn.Linear(feature_size, nb_class),
    )

  def aggregate(self, features_P,features_A,features_F, K):
    relation_pairs_list = list()
    targets_list = list()
    size = int(features_P.shape[0] / K)
    shifts_counter=1
    for index_1 in range(0, size*K, size):
      for index_2 in range(index_1+size, size*K, size):
          for index_3 in range(index_2+size,size*K, size):
            # Using the 'cat' aggregation function by default
            pos1 = features_P[index_1:index_1+size]
            pos2 = features_A[index_2:index_2+size]
            pos3 = features_F[index_3:index_3+size]
            pos_pair1 = torch.cat([pos1,
                                  pos2,pos3], 1)


            # Shuffle without collisions by rolling the mini-batch (negatives)
            neg1 = torch.roll(features_P[index_1:index_1 + size],
                          shifts=shifts_counter, dims=0)
            neg2 = torch.roll(features_F[index_3:index_3 + size],
                              shifts=shifts_counter, dims=0)
            neg_pair=torch.cat([neg1,pos2,neg2],1)


            relation_pairs_list.append(pos_pair1)

            relation_pairs_list.append(neg_pair)


            targets_list.append(torch.ones(size, dtype=torch.float32).cuda())

            targets_list.append(torch.zeros(size, dtype=torch.float32).cuda())


            shifts_counter+=1
            if(shifts_counter>=size):
                shifts_counter=1 # avoid identity pairs
    relation_pairs = torch.cat(relation_pairs_list, 0).cuda()  # K(K-1) * (batch_size, fz*2)
    targets = torch.cat(targets_list, 0).cuda()
    return relation_pairs, targets


  def run_test(self, predict, labels):
      correct = 0
      pred = predict.data.max(1)[1]
      correct = pred.eq(labels.data).cpu().sum()
      return correct, len(labels.data)

  def train(self, tot_epochs, train_loader, train_loader_label, val_loader, test_loader, opt):
    patience = opt.patience
    early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

    optimizer = torch.optim.Adam([
                  {'params': self.backbone.parameters()},
        {'params': self.relation_head.parameters()},
        {'params': self.sup_head.parameters()},
    ], lr=opt.learning_rate)
    c_criterion = nn.CrossEntropyLoss()
    BCE = torch.nn.BCEWithLogitsLoss()

    epoch_max = 0
    acc_max=0
    best_acc=0

    for epoch in range(tot_epochs):
      self.backbone.train()
      self.relation_head.train()
      self.sup_head.train()

      acc_epoch=0
      acc_epoch_cls=0
      loss_epoch=0
      loss_epoch_label=0

      for i, data_labeled in enumerate(train_loader_label):
          optimizer.zero_grad()

          # labeled sample
          (x, target)=data_labeled
          x = x.cuda()
          target = target.cuda()
          output = self.backbone(x)
          output = self.sup_head(output)
          loss_label = c_criterion(output, target)

          loss = loss_label
          loss.backward()
          optimizer.step()

          loss_epoch_label += loss_label.item()

          # estimate the accuracy
          prediction = output.argmax(-1)
          correct = prediction.eq(target.view_as(prediction)).sum()
          accuracy = (100.0 * correct / len(target))
          acc_epoch += accuracy.item()

      for i, (data_augmented, data_P, data_A, data_F, _) in enumerate(train_loader):
        K = len(data_augmented) # tot augmentations
        x_P = torch.cat(data_P, 0).cuda()
        x_A = torch.cat(data_A, 0).cuda()
        x_F = torch.cat(data_F, 0).cuda()

        optimizer.zero_grad()
        # forward pass (backbone)
        features_P = self.backbone(x_P)
        features_A = self.backbone(x_A)
        features_F = self.backbone(x_F)
        # aggregation function
        relation_pairs, targets = self.aggregate(features_P, features_A, features_F, K)

        # forward pass (relation head)
        score = self.relation_head(relation_pairs).squeeze()
        # cross-entropy loss and backward
        loss = BCE(score, targets)
        loss.backward()
        optimizer.step()
        # estimate the accuracy
        predicted = torch.round(torch.sigmoid(score))
        correct = predicted.eq(targets.view_as(predicted)).sum()
        accuracy = (100.0 * correct / float(len(targets)))
        acc_epoch_cls += accuracy.item()
        loss_epoch += loss.item()

      acc_epoch_cls /= len(train_loader)
      loss_epoch /= len(train_loader)
      acc_epoch /= len(train_loader_label)
      loss_epoch_label /= len(train_loader_label)

      if acc_epoch_cls>acc_max:
          acc_max = acc_epoch_cls
          epoch_max = epoch


      acc_vals = list()
      acc_tests = list()
      self.backbone.eval()
      self.sup_head.eval()
      with torch.no_grad():
          for i, (x, target) in enumerate(val_loader):
              x = x.cuda()
              target = target.cuda()

              output = self.backbone(x).detach()
              output = self.sup_head(output)
              # estimate the accuracy
              prediction = output.argmax(-1)
              correct = prediction.eq(target.view_as(prediction)).sum()
              accuracy = (100.0 * correct / len(target))
              acc_vals.append(accuracy.item())

          val_acc = sum(acc_vals) / len(acc_vals)
          if val_acc >= best_acc:
              best_acc = val_acc
              best_epoch = epoch
              for i, (x, target) in enumerate(test_loader):
                  x = x.cuda()
                  target = target.cuda()

                  output = self.backbone(x).detach()
                  output = self.sup_head(output)
                  # estimate the accuracy
                  prediction = output.argmax(-1)
                  correct = prediction.eq(target.view_as(prediction)).sum()
                  accuracy = (100.0 * correct / len(target))
                  acc_tests.append(accuracy.item())

              test_acc = sum(acc_tests) / len(acc_tests)

      print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
          epoch, val_acc, test_acc, best_epoch))
      early_stopping(val_acc, self.backbone)
      if early_stopping.early_stop:
          print("Early stopping")
          break

      if (epoch+1)%opt.save_freq==0:
        print("[INFO] save backbone at epoch {}!".format(epoch))
        torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

      print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, '
            'Max ACC.= {:.1f}%, Max Epoch={}' \
            .format(epoch + 1, opt.model_name, opt.dataset_name,
                    loss_epoch, acc_epoch,acc_epoch_cls, acc_max, epoch_max))
    return test_acc, acc_epoch_cls, best_epoch
class Forecasting(torch.nn.Module):
  """Self-Supervised Relational Reasoning.
  Essential implementation of the method, which uses
  the 'cat' aggregation function (the most effective),
  and can be used with any backbone.
  自我监督的关系推理。
该方法的基本实现，它使用“猫”聚合功能（最有效），并且可以与任何主干一起使用。
  """
  def __init__(self, backbone, feature_size=64, horizon=300, nb_class=3):
    super(Forecasting, self).__init__()
    self.backbone = backbone

    self.forecasting_head = nn.Linear(feature_size, horizon)
    self.sup_head = torch.nn.Sequential(
        torch.nn.Linear(feature_size, nb_class),
    )

  def train(self, tot_epochs, train_loader, train_loader_label, val_loader, test_loader, opt):
    patience = opt.patience
    early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

    optimizer = torch.optim.Adam([
                  {'params': self.backbone.parameters()},
        {'params': self.forecasting_head.parameters()},
        {'params': self.sup_head.parameters()},

    ], lr=opt.learning_rate)
    mse_criterion = nn.MSELoss()
    c_criterion = nn.CrossEntropyLoss()

    acc_epoch = 0
    loss_epoch = 0
    loss_epoch_label = 0

    epoch_max = 0
    acc_max=0
    best_acc=0
    for epoch in range(tot_epochs):
        self.backbone.train()
        self.forecasting_head.train()
        self.sup_head.train()

        acc_epoch=0
        acc_epoch_cls=0
        loss_epoch=0
        for i, data_labeled in enumerate(train_loader_label):
          optimizer.zero_grad()

          # labeled sample
          (x, target)=data_labeled
          x = x.cuda()
          target = target.cuda()
          output = self.backbone(x)
          output = self.sup_head(output)
          loss_label = c_criterion(output, target)

          loss = loss_label
          loss.backward()
          optimizer.step()

          loss_epoch_label += loss_label.item()

          # estimate the accuracy
          prediction = output.argmax(-1)
          correct = prediction.eq(target.view_as(prediction)).sum()
          accuracy = (100.0 * correct / len(target))
          acc_epoch += accuracy.item()


        for i, (data_augmented0, data_label, _) in enumerate(train_loader):
            x_cut0 = torch.cat(data_augmented0, 0).cuda()
            c_label = torch.cat(data_label, 0).cuda()

            optimizer.zero_grad()
            # forward pass (backbone)
            features_cut0 = self.backbone(x_cut0)

            c_output = self.forecasting_head(features_cut0)
            loss = mse_criterion(c_output, c_label)

            loss.backward()
            optimizer.step()
            # estimate the accuracy
            loss_epoch += loss.item()

        loss_epoch /= len(train_loader)
        acc_epoch /= len(train_loader_label)
        loss_epoch_label /= len(train_loader_label)

        acc_vals = list()
        acc_tests = list()
        self.backbone.eval()
        self.sup_head.eval()
        with torch.no_grad():
            for i, (x, target) in enumerate(val_loader):
                x = x.cuda()
                target = target.cuda()

                output = self.backbone(x).detach()
                output = self.sup_head(output)
                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_vals.append(accuracy.item())

            val_acc = sum(acc_vals) / len(acc_vals)
            if val_acc >= best_acc:
                best_acc = val_acc
                best_epoch = epoch
                for i, (x, target) in enumerate(test_loader):
                    x = x.cuda()
                    target = target.cuda()

                    output = self.backbone(x).detach()
                    output = self.sup_head(output)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_tests.append(accuracy.item())

                test_acc = sum(acc_tests) / len(acc_tests)

        print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
            epoch, val_acc, test_acc, best_epoch))
        early_stopping(val_acc, self.backbone)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if (epoch + 1) % opt.save_freq == 0:
            print("[INFO] save backbone at epoch {}!".format(epoch))
            torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

        print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, '
              'Max ACC.= {:.1f}%, Max Epoch={}' \
              .format(epoch + 1, opt.model_name, opt.dataset_name,
                      loss_epoch, acc_epoch, acc_epoch_cls, acc_max, epoch_max))
    return test_acc, acc_epoch_cls, best_epoch
class pseudo(torch.nn.Module):

    def __init__(self, backbone, feature_size=64, nb_class=3):
        super(pseudo, self).__init__()
        self.backbone = backbone
        self.relation_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size * 2, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1))
        self.projict_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1))
        self.sup_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size, nb_class),

        )

    def run_test(self, predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)

    def train(self, tot_epochs, train_loader_label, train_ws_loader, val_loader, test_loader, opt):
        patience = opt.patience
        early_stopping = EarlyStopping(patience, verbose=True,
                                       checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters()},
            {'params': self.relation_head.parameters()},
            {'params': self.sup_head.parameters()},
        ], lr=opt.learning_rate)
        optimizer1 = torch.optim.SGD([
            {'params': self.backbone.parameters()},
            {'params': self.relation_head.parameters()},
            {'params': self.sup_head.parameters()},
        ], lr=opt.learning_rate)
        c_criterion = nn.CrossEntropyLoss()

        epoch_max = 0
        epoch_max1 = 0
        acc_max = 0
        best_acc = 0
        acc_max1 = 0


        def exp_rampup(rampup_length):
            """Exponential rampup from https://arxiv.org/abs/1610.02242"""

            def warpper(epoch):
                if epoch < rampup_length:
                    epoch = np.clip(epoch, 0.0, rampup_length)
                    phase = 1.0 - epoch / rampup_length
                    return float(np.exp(-5.0 * phase * phase))
                else:
                    return 1.0

            return warpper

        for epoch in range(tot_epochs):
            self.backbone.train()
            self.relation_head.train()
            self.sup_head.train()
            losses = misc.AverageMeter()

            acc_epoch = 0

            loss_epoch_label = 0
            acc_epoch_ws = 0
            loss_ws = 0

            for i, data_labeled in enumerate(train_loader_label):
                optimizer.zero_grad()

                (x, target) = data_labeled
                x = x.cuda()
                target = target.long().cuda()
                output = self.backbone(x)
                output = self.sup_head(output)
                loss_label = c_criterion(output, target)

                loss = loss_label
                loss.backward()
                optimizer.step()

                loss_epoch_label += loss_label.item()

                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_epoch += accuracy.item()

            for i, unlabeled in enumerate(train_ws_loader):
                (data_w, label) = unlabeled
                x_w = data_w.cuda()
                label=label.long().cuda()
                output = self.backbone(x_w)
                outputs = self.sup_head(output)

                with torch.no_grad():
                    iter_unlab_pslab = outputs.max(1)[1]
                    iter_unlab_pslab.detach_()
                uloss = c_criterion(outputs,iter_unlab_pslab)
                prediction = outputs.argmax(-1)
                correct = prediction.eq(label.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(label))
                acc_epoch_ws += accuracy.item()
                loss = uloss
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                losses.update(loss.item())
                loss_ws += loss.item()

            acc_epoch_ws /= len(train_ws_loader)
            loss_ws /= len(train_ws_loader)

            acc_epoch /= len(train_loader_label)
            loss_epoch_label /= len(train_loader_label)


            if acc_epoch_ws > acc_max1:
                acc_max1 = acc_epoch_ws
                epoch_max1 = epoch

            acc_vals = list()
            acc_tests = list()
            self.backbone.eval()
            self.sup_head.eval()
            with torch.no_grad():
                for i, (x, target) in enumerate(val_loader):
                    x = x.cuda()
                    target = target.cuda()

                    output = self.backbone(x).detach()
                    output = self.sup_head(output)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_vals.append(accuracy.item())

                val_acc = sum(acc_vals) / len(acc_vals)
                if val_acc >= best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    for i, (x, target) in enumerate(test_loader):
                        x = x.cuda()
                        target = target.cuda()

                        output = self.backbone(x).detach()
                        output = self.sup_head(output)
                        # estimate the accuracy
                        prediction = output.argmax(-1)
                        correct = prediction.eq(target.view_as(prediction)).sum()
                        accuracy = (100.0 * correct / len(target))
                        acc_tests.append(accuracy.item())

                    test_acc = sum(acc_tests) / len(acc_tests)

            print('[Test-{}] Val ACC:{:.2f}%,Best val ACC.: {:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
                epoch, val_acc, best_acc, test_acc, best_epoch))
            early_stopping(val_acc, self.backbone)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if (epoch + 1) % opt.save_freq == 0:
                print("[INFO] save backbone at epoch {}!".format(epoch))
                torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

            print('Epoch [{}][{}][{}] loss= {:.5f};loss= {:.5f}; Epoch ACC.= {:.2f}%,Max ACC.= {:.2f}%,Max Epoch={}' \
                  .format(epoch + 1, opt.model_name, opt.dataset_name,
                          loss_ws, loss_epoch_label, acc_epoch, acc_max1, epoch_max1))
        return test_acc, val_acc, best_acc, best_epoch
class pi(torch.nn.Module):

    def __init__(self, backbone, feature_size=64, nb_class=3):
        super(pi, self).__init__()
        self.backbone = backbone

        self.relation_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size * 2, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1))
        self.projict_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1))
        self.sup_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size, nb_class),

        )

    def run_test(self, predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)

    def train(self, tot_epochs, train_loader_label, train_ws_loader, val_loader, test_loader, opt):
        self.usp_weight = opt.usp_weight
        self.rampup = exp_rampup(opt.weight_rampup)


        self.epoch = 0
        patience = opt.patience

        early_stopping = EarlyStopping(patience, verbose=True,
                                       checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters()},
            {'params': self.relation_head.parameters()},
            {'params': self.sup_head.parameters()},
        ], lr=opt.learning_rate)
        optimizer1 = torch.optim.SGD([
            {'params': self.backbone.parameters()},
            {'params': self.relation_head.parameters()},
            {'params': self.sup_head.parameters()},
        ], lr=opt.learning_rate)
        c_criterion = nn.CrossEntropyLoss()

        epoch_max = 0
        epoch_max1 = 0
        acc_max = 0
        best_acc = 0
        acc_max1 = 0

        def mse_with_softmax(logit1, logit2):
            assert logit1.size() == logit2.size()
            return F.mse_loss(F.softmax(logit1, 1), F.softmax(logit2, 1))

        for epoch in range(tot_epochs):
            self.backbone.train()
            self.relation_head.train()
            self.sup_head.train()
            losses = misc.AverageMeter()

            acc_epoch = 0

            loss_epoch_label = 0
            acc_epoch_ws = 0
            loss_ws = 0
            for i, data_labeled in enumerate(train_loader_label):
                optimizer.zero_grad()
                (x, target) = data_labeled
                x = x.cuda()
                target = target.cuda()
                output = self.backbone(x)
                output = self.sup_head(output)
                loss_label = c_criterion(output, target)

                loss = loss_label
                loss.backward()
                optimizer.step()
                loss_epoch_label += loss_label.item()

                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_epoch += accuracy.item()

            for i,unlabeled in enumerate(train_ws_loader):
                (data,data_w, label) = unlabeled
                label = label.cuda()
                x_notransform = data.cuda()
                pi_outputs1 = self.backbone(x_notransform)
                pi_output1 = self.sup_head(pi_outputs1)
                with torch.no_grad():
                    x_w = data_w.cuda()
                    pi_outputs = self.backbone(x_w)
                    pi_output  =self.sup_head(pi_outputs)
                    pi_output = pi_output.detach()
                cons_loss = mse_with_softmax(pi_output1, pi_output)
                cons_loss *= self.rampup(self.epoch) * self.usp_weight
                prediction = pi_output1.argmax(-1)
                correct = prediction.eq(label.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(pi_outputs))
                acc_epoch_ws += accuracy.item()
                loss1 = cons_loss
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()
                losses.update(loss1.item())
                loss_ws += loss1.item()

            acc_epoch_ws /= len(train_ws_loader)
            loss_ws /= len(train_ws_loader)


            acc_epoch /= len(train_loader_label)
            loss_epoch_label /= len(train_loader_label)

            if acc_epoch_ws > acc_max1:
                acc_max1 = acc_epoch_ws
                epoch_max1 = epoch

            acc_vals = list()
            acc_tests = list()
            self.backbone.eval()
            self.sup_head.eval()
            with torch.no_grad():
                for i, (x, target) in enumerate(val_loader):
                    x = x.cuda()
                    target = target.cuda()

                    output = self.backbone(x).detach()
                    output = self.sup_head(output)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_vals.append(accuracy.item())

                val_acc = sum(acc_vals) / len(acc_vals)
                if val_acc >= best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    for i, (x, target) in enumerate(test_loader):
                        x = x.cuda()
                        target = target.cuda()

                        output = self.backbone(x).detach()
                        output = self.sup_head(output)
                        # estimate the accuracy
                        prediction = output.argmax(-1)
                        correct = prediction.eq(target.view_as(prediction)).sum()
                        accuracy = (100.0 * correct / len(target))
                        acc_tests.append(accuracy.item())

                    test_acc = sum(acc_tests) / len(acc_tests)

            print('[Test-{}] Val ACC:{:.2f}%,Best val ACC.: {:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
                epoch, val_acc, best_acc, test_acc, best_epoch))
            early_stopping(val_acc, self.backbone)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if (epoch + 1) % opt.save_freq == 0:
                print("[INFO] save backbone at epoch {}!".format(epoch))
                torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

            print('Epoch [{}][{}][{}] loss= {:.5f};loss= {:.5f}; Epoch ACC.= {:.2f}%,Max ACC.= {:.2f}%,Max Epoch={}' \
                  .format(epoch + 1, opt.model_name, opt.dataset_name,
                          loss_ws, loss_epoch_label, acc_epoch, acc_max1, epoch_max1))
        return test_acc, val_acc, best_acc, best_epoch

