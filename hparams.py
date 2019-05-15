import argparse



parser = argparse.ArgumentParser(description='Speech domain adpataion')


parser.add_argument('--cuda', type=bool, default=True, help='Set cuda usage')
parser.add_argument('--spec_opt', type=str, default='mel', help='Linear spec or mel spec')
parser.add_argument('--epoch_size', type=int, default=50000, help='Set epoch size')
parser.add_argument('--batch_size', type=int, default=16, help='Set batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Set learning rate for optimizer')
parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--dataset_root', type=str, default= '../dataset/feat/train/', help ='Root diection of Dataset')

parser.add_argument('--list_dir_train_A_2d', type =str, default = './etc/Train_tr05_real_noisy_list.csv')
parser.add_argument('--list_dir_train_B_2d', type =str, default = './etc/Train_tr05_orig_clean_list.csv')

parser.add_argument('--speakers', type=str, default = ['noisy', 'clean'])
parser.add_argument('--task_name', type=str, default = 'chime4')

parser.add_argument('--iteration_num',type=int, default=100000)


parser.add_argument('--log_interval', type=int, default=200, help='Print loss values every log_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=10000, help='Save models every model_save_interval iterations.')



def get_hparams():
	config  = parser.parse_args()

	return config

