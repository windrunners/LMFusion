
class args():

	
	# training args
	epochs = 4 #"number of training epochs, default is 2"
	batch_size = 1 #"batch size for training, default is 4"
	dataset = "MSCOCO 2014"
	HEIGHT = 256
	WIDTH = 256

	save_model_dir = "models" #"path to folder where trained model will be saved."
	save_loss_dir = "models/loss"  # "path to folder where trained model will be saved."

	image_size = 256 #"size of training images, default is 256 X 256"
	cuda = 0 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"
	ssim_weight = [1, 10, 100, 1000, 10000]
	grd_weight = [0.01, 0.1, 1, 10, 100, 1000, 10000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 1e-4 #"learning rate, default is 0.001"
	lr_light = 1e-4  # "learning rate, default is 0.001"
	log_interval = 5 #"number of images after which the training loss is logged, default is 500"
	resume = None
	resume_auto_en = None
	resume_auto_de = None
	resume_auto_fn = None

	# for test Final.model
	model_path_gray = "./models/LMFusion_gray.model"



