import argparse
from configs import get_cfg
from util.net import init_training
from util.util import run_pre, init_checkpoint
from trainer import get_trainer
import warnings
warnings.filterwarnings("ignore")


def main():
	parser = argparse.ArgumentParser()
	# parser.add_argument('-c', '--cfg_path', default='configs/rd_mvtec_debug.py')
	# parser.add_argument('-c', '--cfg_path', default='configs/invad_mvtec_debug.py')
	parser.add_argument('-c', '--cfg_path', default='configs/myad/myad_mvtec.py')
	parser.add_argument('-m', '--mode', default='train', choices=['train', 'test'])
	parser.add_argument('--sleep', type=int, default=-1)
	parser.add_argument('--memory', type=int, default=-1)
	parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
	parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
	parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER,)
	# 配置cfg的参数cfg_terminal
	cfg_terminal = parser.parse_args()
	# 获取cfg
	cfg = get_cfg(cfg_terminal)
	# 对cfg进行预处理，设置睡眠时间、等待GPU内存
	run_pre(cfg)
	# 初始化训练环境：
	# 配置 GPU 环境和 CuDNN 设置，确保 GPU 可用性并优化训练性能。
	# 根据环境变量初始化分布式训练的配置，包括进程数、GPU 数量和节点数等。
	# 初始化随机种子，确保训练的可重复性。
	# 配置数据加载器的批量大小和工作线程数，支持分布式训练环境
	init_training(cfg)
	# 初始化模型的检查点目录、加载已有的检查点（如果指定了），并为新的训练任务创建新的日志目录
	init_checkpoint(cfg)
	trainer = get_trainer(cfg)
	trainer.run()


if __name__ == '__main__':
	main()
