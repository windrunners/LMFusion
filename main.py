from models import *
import time
from torch.backends import cudnn
import torch,warnings
from torch import nn
warnings.filterwarnings('ignore')
from option import model_name, log_dir
from data_utils import *
from torchvision.models import vgg16
print('log_dir :',log_dir)
print('model_name:', model_name)

import torch
from net import LMFusion_net
from args_fusion import args
import numpy as np
import os
import pytorch_msssim
from piecewise_optimizer import piecewise



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, input_nc, output_nc):

	LMFusion_model = LMFusion_net(input_nc, output_nc)
	LMFusion_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in LMFusion_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(LMFusion_model._get_name(), para * type_size / 1000 / 1000))

	LMFusion_model.eval()

	return LMFusion_model

def _generate_fusion_image(model, img1):
	# encoder
	en_r = model.encoder(img1)
	return en_r

model_path = args.model_path_gray
model = load_model(model_path, 1, 1)


models_={
	'LMF': LMF(),
}
loaders_={
	'its_train':ITS_train_loader,
	'its_test':ITS_test_loader,
	'ots_train':OTS_train_loader,
	'ots_test':OTS_test_loader
}
start_time=time.time()
T=opt.steps
ssim_loss = pytorch_msssim.msssim
mse_loss = torch.nn.MSELoss()
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train(net,loader_train,loader_test,optim,criterion):
	losses = []
	loss_tests = []
	start_step = 0
	max_ssim = 0
	max_psnr = 0
	min_loss_test = 1
	ssims = []
	psnrs = []
	if opt.resume and os.path.exists(opt.model_dir):
		print(f'resume from {opt.model_dir}')
		ckp = torch.load(opt.model_dir)
		losses = ckp['losses']
		net.load_state_dict(ckp['model'])
		start_step = ckp['step']
		max_ssim = ckp['max_ssim']
		max_psnr = ckp['max_psnr']
		psnrs = ckp['psnrs']
		ssims = ckp['ssims']
		print(f'start_step:{start_step} start training ---')
	else:
		print('train from scratch *** ')
	for step in range(start_step+1, opt.steps+1):
		net.train()
		lr=opt.lr
		if not opt.no_lr_sche:
			lr=lr_schedule_cosdecay(step,T)
			for param_group in optim.param_groups:
				param_group["lr"] = lr  
		x, y=next(iter(loader_train))
		# 这里输出的是loader_train中的一个batch。loader_train是一个iterable object，通过iter()得到iterator，用next()可以访问这个iterator的下一个元素
		# x=x.to(opt.device);y=y.to(opt.device)

		# n11, c11, h11, w11 = y.shape
		# y = y.reshape(n11*c11, h11, w11)
		# img_y = y.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

		# img_y = np.squeeze(img_y)
		# plt.imshow(img_y, cmap='gray')   # plt.imshow(img, cmap='gray')
		# plt.show()

		# n12, c12, h12, w12 = x.shape
		# x = x.reshape(n12*c12, h12, w12)
		# img_x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

		# img_x = np.squeeze(img_x)
		# plt.imshow(img_x, cmap='gray')   # plt.imshow(img, cmap='gray')
		# plt.show()


		x1 = _generate_fusion_image(model, x)    #
		y1 = _generate_fusion_image(model, y)    #
		#   x11 = type(x)
		#   y11 = type(x1)
		x2 = torch.stack(x1)
		y2 = torch.stack(y1)
		b1, n1, c1, h1, w1 = x2.shape
		b2, n2, c2, h2, w2 = y2.shape
		x3 = x2.reshape(b1, n1*c1, h1, w1)
		y3 = y2.reshape(b2, n2*c2, h2, w1)

		out = net(x3)

		# loss = criterion[0](out, y3)
		loss = criterion[0](out, y3) + (1-ssim_loss(out, y3, normalize=True))
		if opt.perloss:
			loss2 = criterion[1](out, y3)
			loss = loss+0.04*loss2

		loss.backward()  # back propagation, calculate the current gradient

		optim.step()  # Update the network parameters according to the gradient
		optim.zero_grad()  # clear the existing gradient, free up space
		losses.append(loss.item())  # .item(): return a more accurate floating-point number
		print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)



		if step % opt.eval_step == 0:
			with torch.no_grad():  # backpropagation is not used
				ssim_eval, psnr_eval, loss_test_eval = test(net, loader_test, max_psnr, max_ssim, step)

			print(f'\nstep :{step}|loss_test:{loss_test_eval:.4f} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)
			loss_tests.append(loss_test_eval)
			max_psnr = max(max_psnr, psnr_eval)

			if loss_test_eval < min_loss_test and ssim_eval > max_ssim:
				max_ssim = max(max_ssim, ssim_eval)
				min_loss_test = min(min_loss_test, loss_test_eval)
				torch.save({
					'step': step,
					'max_psnr': max_psnr,
					'max_ssim': max_ssim,
					'ssims': ssims,
					'psnrs': psnrs,
					'losses': losses,
					'model': net.state_dict()
				}, opt.model_dir)
				print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

				# save_model_path = os.path.join('models', 'LMFUsion_gray1.model')
				# torch.save(model.state_dict(), save_model_path)

	np.save(f'./numpy_files/{model_name}_{opt.steps}_{time.strftime("%Y-%m-%d %H%M%S")}_losses.npy', losses)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_{time.strftime("%Y-%m-%d %H%M%S")}_ssims.npy', ssims)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_{time.strftime("%Y-%m-%d %H%M%S")}_psnrs.npy', psnrs)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_{time.strftime("%Y-%m-%d %H%M%S")}_test_loss.npy', loss_tests)






def test(net, loader_test, max_psnr, max_ssim, step):
	net.eval()
	torch.cuda.empty_cache()  # release the video memory
	loss_tests = []
	ssims=[]
	psnrs=[]

	for i, (inputs, targets) in enumerate(loader_test):
		inputs=inputs.to(opt.device); targets=targets.to(opt.device)

		x_1 = _generate_fusion_image(model, inputs)
		y_1 = _generate_fusion_image(model, targets)
		x_2 = torch.stack(x_1)
		y_2 = torch.stack(y_1)
		b_1, n_1, c_1, h_1, w_1 = x_2.shape
		b_2, n_2, c_2, h_2, w_2 = y_2.shape
		inputs_ = x_2.reshape(b_1, n_1 * c_1, h_1, w_1)
		targets_ = y_2.reshape(b_2, n_2 * c_2, h_2, w_1)

		pred=net(inputs_)

		ssim1 = ssim(pred, targets_).item()
		psnr1 = psnr(pred, targets_)
		loss_test1 = criterion[0](pred, targets_).item()
		loss_tests.append(loss_test1)
		ssims.append(ssim1)
		psnrs.append(psnr1)

	return np.mean(ssims), np.mean(psnrs), np.mean(loss_tests)



if __name__ == "__main__":
	loader_train=loaders_[opt.trainset]
	loader_test=loaders_[opt.testset]
	net = models_[opt.net]
	net = net.to(opt.device)
	if opt.device=='cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark=True
	criterion = []
	criterion.append(nn.L1Loss().to(opt.device))
	if opt.perloss:
			vgg_model = vgg16(pretrained=True).features[:16]
			vgg_model = vgg_model.to(opt.device)
			for param in vgg_model.parameters():
				param.requires_grad = False
			criterion.append(PerLoss(vgg_model).to(opt.device))

	optimizer = piecewise(net.parameters(), lr=opt.lr, weight_decay=1e-3)

	optimizer.zero_grad()
	train(net, loader_train, loader_test, optimizer, criterion)

