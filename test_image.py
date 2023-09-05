
import torch
from net import LMFusion_net
import utils
from args_fusion import args
import numpy as np
from models import *
import os
import torchvision.transforms as tfs
import cv2
import imageio
import warnings
warnings.filterwarnings("ignore")

abs=os.getcwd()+'/'   # return the current working directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = abs + f'trained_models/its_train_LMF.pk'
ckp = torch.load(model_dir, map_location=device)
net_ = LMF()
net_.load_state_dict(ckp['model'], strict = False)
net_.eval()


def load_model(path, input_nc, output_nc):

	LMFusion_model = LMFusion_net(input_nc, output_nc)
	LMFusion_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in LMFusion_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(LMFusion_model._get_name(), para * type_size / 1000 / 1000))

	LMFusion_model.eval()
	# LMFusion_model.to(device)

	return LMFusion_model


def _generate_fusion_image(model, strategy_type, img1, img2, p_type):

	img1 = tfs.Compose([
		tfs.ToTensor(),
		# tfs.Normalize(mean=[0.4], std=[0.3])    # mean=[0.4, 0.4, 0.4], std=[0.35, 0.35, 0.35]
	])(img1)[None, ::]

	img2 = tfs.Compose([
		tfs.ToTensor(),
		# tfs.Normalize(mean=[0.4], std=[0.3])
	])(img2)[None, ::]



	# encoder
	en_r = model.encoder(img1)   # list （1,65,256,256）
	en_v = model.encoder(img2)	 # list （1,65,256,256）


	x_2 = torch.stack(en_r)  # tensor （1,1,65,256,256）
	y_2 = torch.stack(en_v)  # tensor （1,1,65,256,256）
	b1, n1, c1, h1, w1 = x_2.shape
	b2, n2, c2, h2, w2 = y_2.shape
	x_3 = x_2.reshape(b1, n1 * c1, h1, w1)  # tensor （1,65,256,256）
	y_3 = y_2.reshape(b2, n2 * c2, h2, w2)  # tensor （1,65,256,256）

	en_r2 = net_(x_3)  # tensor （1,65,256,256）
	en_v2 = net_(y_3)  # tensor （1,65,256,256）

	en_r[0] = en_r2
	en_v[0] = en_v2

	# fusion: hybrid, channel and spatial
	# f = model.fusion(en_r, en_v, p_type)

	# fusion: addition
	f = model.fusion1(en_r, en_v)

	# fusion: composite attention
	# f = model.fusion2(en_r, en_v, p_type)

	# decoder
	img_fusion = model.decoder(f)
	return img_fusion[0]


def run_demo(model, im_num, shape1, img1, img1_cb, img1_cr, img2, img2_cb, img2_cr, output_path_root, fusion_type, network_type, strategy_type, ssim_weight_str, mode, p_type):

	img_fusion = _generate_fusion_image(model, strategy_type, img1, img2, p_type)

	img_fusion = tfs.Compose([
		tfs.Normalize(mean=[0.4], std=[0.3])
	])(img_fusion)

	img_fusion1 = img_fusion.detach().numpy()
	output = img_fusion1[0, 0, :, :]

	if len(shape1) > 2:
		output = ycbcr2rgb(output, (img1_cb + img2_cb)/2, (img1_cr + img2_cr)/2)


	file_name = str(im_num) + '_LMFusion.jpg'
	output_path = output_path_root + file_name
	imageio.imsave(output_path, output)

	# img = cv2.imread(output_path)
	# img = aug(img)
	# imageio.imsave(output_path, img)




def compute(img, min_percentile, max_percentile):
	"""计算分位点，目的是去掉图1的直方图两头的异常情况"""
	max_percentile_pixel = np.percentile(img, max_percentile)
	min_percentile_pixel = np.percentile(img, min_percentile)
	return max_percentile_pixel, min_percentile_pixel

def aug(src):
	"""图像亮度增强"""
	if get_lightness(src) > 130:
		print("图片亮度足够，不做增强")
	# 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
	# 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。
	max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)
	# 去掉分位值区间之外的值
	src[src >= max_percentile_pixel] = max_percentile_pixel
	src[src <= min_percentile_pixel] = min_percentile_pixel
	# 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
	out = np.zeros(src.shape, src.dtype)
	cv2.normalize(src, out, 255*0.1, 255*0.9, cv2.NORM_MINMAX)
	return out

def get_lightness(src):
	# 计算亮度
	hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
	lightness = hsv_image[:, :, 2].mean()
	return lightness




def vision_features(feature_maps, img_type):
	count = 0
	for features in feature_maps:
		count += 1
		for index in range(features.size(1)):
			file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_channel_' + str(index) + '.png'
			output_path = 'outputs/feature_maps/' + file_name
			map = features[:, index, :, :].view(1,1,features.size(2),features.size(3))
			map = map*255
			# save images
			utils.save_image_test(map, output_path)

def rgb2ycbcr(img_rgb):
	R = img_rgb[:, :, 0]
	G = img_rgb[:, :, 1]
	B = img_rgb[:, :, 2]
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
	Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
	return Y, Cb, Cr


def ycbcr2rgb(Y, Cb, Cr):
	R = Y + 1.402 * (Cr - 128 / 255.0)
	G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
	B = Y + 1.772 * (Cb - 128 / 255.0)
	R = np.expand_dims(R, axis=-1)
	G = np.expand_dims(G, axis=-1)
	B = np.expand_dims(B, axis=-1)
	return np.concatenate([R, G, B], axis=-1)



def main():

	network_type = 'TUfusion'
	strategy_type_list = ['addition', 'attention_weight']

	output_path = './outputs/'
	strategy_type = strategy_type_list[0]
	fusion_type = ['attention_max']
	p_type = fusion_type[0]

	if os.path.exists(output_path) is False:
		os.mkdir(output_path)

	in_c = 1
	out_c = in_c
	mode = 'L'
	model_path = args.model_path_gray

	with torch.no_grad():
		ssim_weight_str = args.ssim_path[2]
		model = load_model(model_path, in_c, out_c)

	abs = os.getcwd() + '/'  # return the current working directory # 移
	img_dir1 = abs + "test_fusion_imgs/modality1/"
	img_dir2 = abs + "test_fusion_imgs/modality2/"

	for im in os.listdir(img_dir1):
		print(f'\r {im}', end='', flush=True)
		img1 = imageio.imread(img_dir1 + im.split('/',)[-1], pilmode='RGB') / 255.0
		img2 = imageio.imread(img_dir2 + im.split('/',)[-1], pilmode='RGB') / 255.0

		im_num = im[:-4]

		shape1 = img1.shape
		shape2 = img2.shape
		h1 = shape1[0]
		w1 = shape1[1]
		h2 = shape2[0]
		w2 = shape2[1]

		img1, img1_cb, img1_cr = rgb2ycbcr(img1)
		img2, img2_cb, img2_cr = rgb2ycbcr(img2)

		if h1 == h2 and w1 == w2:
			img1 = img1.reshape([h1, w1, 1])
			img2 = img2.reshape([h1, w1, 1])
		else:
			print('please adjust data dimension')
		img1 = img1.astype(np.float32)
		img2 = img2.astype(np.float32)

		run_demo(model, im_num, shape1, img1, img1_cb, img1_cr, img2, img2_cb, img2_cr, output_path, fusion_type, network_type, strategy_type, ssim_weight_str, mode, p_type)
	print('Done......')

if __name__ == '__main__':
	main()
