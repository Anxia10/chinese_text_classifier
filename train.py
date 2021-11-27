import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def train(config,model,train_iter,dev_iter,test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    # writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i,(trains,labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()  # 每次反向传播时先将模型梯度清零
            loss = F.cross_entropy(outputs, labels) # 使用交叉熵作为损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 参数更新
            if total_batch % 100 == 0:  # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data,1)


