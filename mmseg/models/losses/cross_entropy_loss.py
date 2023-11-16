import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
import cv2
import numpy as np
def connect_loss(pred1,
label,label_np):
  #active and normalize
  label=label*1.0
  pred=pred1.clone()
  #pred=F.relu(pred)
  pred=F.softmax(pred, dim=0)

  contours, hierarchy = cv2.findContours(label_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  loss=0
  gt_mean=0
  gt_var=0
  pred_mean=0
  pred_var=0
  c=0.1
  
  for j in range(len(contours)):
    for i in range(contours[j].shape[0]):
      c+=1
      x,y=contours[j][i][0]
      #print(x,y)
      #x_end=x+1
      #x_start=x
      #if x_end+1<=label_np.shape[1]:
      #  x_end=x_end+1
      #if x_start-1 >= 0:
      #  x_start=x_start-1
      #y_end=y+1
      #y_start=y
      #if y_end+1<=label_np.shape[0]:
      #  y_end=y_end+1
      #if y_start-1 >= 0:
      #  y_start=y_start-1
      #pred_edge=pred[1,y,x]
      #gt_edge=label[y,x]

      #pred_var=pred[1,y_start:y_end,x_start:x_end].var()

      #print('p:',pred[:,y_start:y_end,x_start:x_end])

      #gt_var=label[y_start:y_end,x_start:x_end].var()
      #print('l:',label[y_start:y_end,x_start:x_end])

      #loss+=(pred_edge-gt_edge)**2+(gt_mean-pred_mean)**2+(gt_var-pred_var)**2
      loss += F.cross_entropy(
        pred[:,:,y:y+1,x:x+1],
        label[:,y:y+1,x:x+1],
        weight=None,
        reduction='none',
        ignore_index=-100)
  return loss/c
  
def con_loss(pred,label):
  #内轮廓
  #print('label.max():',label.max())
  index_T=label==2
  index_F=label[:,:-1,:]!=2
  index_T_1=label[:,:-1,:]==2
  label_true=label.clone()
  label_true[index_T]=1.
  #print('label_true.max():',label_true.max())
  pred_logit=pred.clone()
  #get class
  out1=F.softmax(pred_logit,dim=1)
  out=out1[:,1,:,:]
  pred1=out[:,:-1,:]-out[:,1:,:]
  label1=label_true[:,:-1,:]-label_true[:,1:,:]
  loss1=F.l1_loss(pred1,label1, reduction='none')
  loss1[index_F]=loss1[index_F]*0
  loss1=loss1/(index_T_1.sum()+0.001)*511*512*4
  
  index_T=label==2
  index_F=label[:,:,:-1]!=2
  index_T_1=label[:,:,:-1]==2
  label_true=label.clone()
  label_true[index_T]=1.
  #print('label_true.max():',label_true.max())
  pred_logit=pred.clone()
  #get class
  out1=F.softmax(pred_logit,dim=1)
  out=out1[:,1,:,:]
  pred1=out[:,:,:-1]-out[:,:,1:]
  label1=label_true[:,:,:-1]-label_true[:,:,1:]
  loss2=F.l1_loss(pred1,label1, reduction='none')
  loss2[index_F]=loss2[index_F]*0
  loss2=loss2/(index_T_1.sum()+0.001)*511*512*4
  
  
  index_T=label==2
  index_F=label[:,1:,:]!=2
  index_T_1=label[:,1:,:]==2
  label_true=label.clone()
  label_true[index_T]=1.
  #print('label_true.max():',label_true.max())
  pred_logit=pred.clone()
  #get class
  out1=F.softmax(pred_logit,dim=1)
  out=out1[:,1,:,:]
  pred1=out[:,1:,:]-out[:,:-1,:]
  label1=label_true[:,1:,:]-label_true[:,:-1,:]
  loss3=F.l1_loss(pred1,label1, reduction='none')
  loss3[index_F]=loss3[index_F]*0
  loss3=loss3/(index_T_1.sum()+0.001)*511*512*4
  
  index_T=label==2
  index_F=label[:,:,1:]!=2
  index_T_1=label[:,:,1:]==2
  label_true=label.clone()
  label_true[index_T]=1.
  #print('label_true.max():',label_true.max())
  pred_logit=pred.clone()
  #get class
  out1=F.softmax(pred_logit,dim=1)
  out=out1[:,1,:,:]
  pred1=out[:,:,1:]-out[:,:,:-1]
  label1=label_true[:,:,1:]-label_true[:,:,:-1]
  loss4=F.l1_loss(pred1,label1, reduction='none')
  loss4[index_F]=loss4[index_F]*0
  loss4=loss4/(index_T_1.sum()+0.001)*511*512*4
  return loss1, loss2, loss3,loss4

def sobel(pred,label,counter):
    #label shape=(4,512,512)
    #print('max:',pred.max())
    beta=1000.0
    pred=pred*beta
    out1=F.softmax(pred,dim=1)
    pred_img=out1[:,1:2,:,:]*1.0
    '''
    out1= torch.argmax(pred,dim=1)
    out1=out1.reshape((4,1,512,512))
    pred_img=out1[:,:,:,:]*1.0
    '''
    label_img=(label*1.0).reshape((4,1,512,512))
    
    conv1 = nn.Conv2d(1,1,3,padding=1,bias=False)
    sobel_kernel = np.array([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],dtype='float32')
    #sobel_kernel = sobel_kernel.reshape((1,1,3,3)) 
    conv1.weight.data = torch.from_numpy(sobel_kernel).cuda()
    #print(conv1.weight.data)
    
    conv2 = nn.Conv2d(1,1,3,padding=1,bias=False)
    sobel_kernel_2 = np.array([[[[-1,-2,-1],[0,0,0],[1,2,1]]]],dtype='float32')
    #sobel_kernel_2 = sobel_kernel.reshape((1,1,3,3)) 
    conv2.weight.data = torch.from_numpy(sobel_kernel_2).cuda()
    #print(conv2.weight.data)
    #print('shu:',(conv1(label_img)**2).max())
    #print('heng:',(conv1(label_img)**2).max())
    
    pred_edge=(conv1(pred_img)**2+conv2(pred_img)**2)/((conv1(pred_img)**2+conv2(pred_img)**2).max()+0.001)
    label_edge=(conv1(label_img)**2+conv2(label_img)**2)/((conv1(label_img)**2+conv2(label_img)**2).max()+0.001)
    #print((conv1(pred_img)**2+conv2(pred_img)**2).max())
    #print(((conv1(label_img)**2+conv2(label_img)**2)).max())
    #print(pred_img)
    #print(pred_edge)
    #print(label_edge)
    '''
    if counter % 10000 == 999:
      print('out1:',out1)
      print('pred_img:',pred_img)
      draw(label_img,'label')
      draw(label_edge,'label_edge')
      draw(pred_img,'pred')
      draw(pred_edge,'pred_edge')
      import pdb;pdb.set_trace()
    '''
    loss=(pred_edge-label_edge)**2
    #print('loss:',loss)
    #loss=-(1-label_edge)*torch.log(1-pred_edge)-label_edge*torch.log(pred_edge)
    
    return loss.squeeze()
def draw(pred,i):
    print(pred.shape)
    img=pred.cpu().detach().numpy()*255
    cv2.imwrite('/mnt/linjie/SeMask-FPN/draw/semantic_map/'+str(i)+'(1).png',img[1,0,:,:])
    cv2.imwrite('/mnt/linjie/SeMask-FPN/draw/semantic_map/'+str(i)+'(2).png',img[2,0,:,:])
    cv2.imwrite('/mnt/linjie/SeMask-FPN/draw/semantic_map/'+str(i)+'(3).png',img[3,0,:,:])
    cv2.imwrite('/mnt/linjie/SeMask-FPN/draw/semantic_map/'+str(i)+'(0).png',img[0,0,:,:])
    
def binary(pred,label):
    #label shape=(4,512,512)
    a=np.random.randint(0,100)
    
    out1=F.softmax(pred,dim=1)
    if a==50:
      print(out1)
    pred_img=out1[:,1:2,:,:]*1.0
    label_img=(label*1.0).reshape((4,1,512,512))
    conv1 = nn.Conv2d(1,1,3,padding=1,bias=False)
    sobel_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1,1,3,3)) 
    #sobel_kernel = torch.LongTensor([[[[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]]]).cuda()
    conv1.weight.data = torch.from_numpy(sobel_kernel).cuda()
    #print(conv1.weight.data)
    pred_edge=conv1(pred_img)
    label_edge=conv1(label_img)
    loss=(pred_edge-label_edge)**2
    return loss.squeeze()
    
def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  counter=0):
    """The wrapper function for :func:`F.cross_entropy`"""
    
    index_2_T=label==2
    index_2_F=label!=2
    index_3_T=label==3
    index_3_F=label!=3
    index_F=index_2_F*index_3_F
    label_true=label.clone()
    label_true[index_2_T]=0
    label_true[index_3_T]=1
    
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)
    
    #print(loss.shape)
    '''
    loss1=loss.clone()
    loss1[index_F]=loss1[index_F]*0
    loss1=loss1*(512*512*4)/(index_2_T.sum()+index_3_T.sum()+0.001)
    '''
    
    #print(loss1.shape)
    #loss1=sobel(pred,label_true,counter)
    #loss+=0.1*loss1
    
    #print('label:',label.max())
    #print('label_true:',label_true.max())
    #import pdb;pdb.set_trace()
    # apply weights and do the reduction
    
    
    
    #loss1,loss2,loss3,loss4 =con_loss(pred,label)
    
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
   #loss1 = weight_reduce_loss(
        #loss1, weight=weight, reduction=reduction, avg_factor=avg_factor)
    #print(loss1)
    '''
    loss1 = weight_reduce_loss(
        loss1, weight=weight, reduction=reduction, avg_factor=avg_factor)
    loss2 = weight_reduce_loss(
        loss2, weight=weight, reduction=reduction, avg_factor=avg_factor)
    loss3 = weight_reduce_loss(
        loss3, weight=weight, reduction=reduction, avg_factor=avg_factor)
    loss4 = weight_reduce_loss(
        loss4, weight=weight, reduction=reduction, avg_factor=avg_factor)
    '''

    return loss
    #return c_loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=255):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored. Default: 255

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        label, weight = _expand_onehot_labels(label, weight, pred.shape,
                                              ignore_index)

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]

def focal_loss(preds=None,
                labels=None,
                weight=None,
                class_weight=None,
                reduction='mean',
                avg_factor=None,
                ignore_index=-100,
                a=0.25,
                gamma=2,
                num_classes = 2, size_average=True):
                

        alpha = torch.zeros(num_classes)
        alpha[0] += a
        alpha[1:] += (1-a) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        preds = preds.view(-1,preds.size(-1))        
        alpha = alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)        
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )        
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        alpha = alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss

@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.counter=0

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
            #self.cls_criterion =focal_loss
        

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        counter=self.counter
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            counter=counter,
            **kwargs)
        self.counter+=1
        #print(self.counter)
        return loss_cls
        
        
        
# -*- coding: utf-8 -*-
# @Author  : LG
@LOSSES.register_module()
class focal_loss(nn.Module):    
    def __init__(self, alpha=0.25, gamma=2, num_classes = 2, size_average=True):
        
        
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma
        
    def forward(self, preds, labels):
        #import pdb;pdb.set_trace()
        # assert preds.dim()==2 and labels.dim()==1        
        preds = preds.view(-1,preds.size(-1))        
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)        
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )        
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss
