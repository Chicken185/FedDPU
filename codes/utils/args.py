from argparse import ArgumentParser
from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')

    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')

    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')

    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')

    # Designed for FL_PU
    parser.add_argument('--pos_class_list', type=int, nargs='+', default=[0], 
                        help='List of positive classes (e.g., --pos_class_list 0 1)')
    
    # 2. 公共探针集大小
    parser.add_argument('--public_size', type=int, default=1000, 
                        help='Size of the public probe dataset (D_pub)')
    
    # 3. 预热轮次
    parser.add_argument('--Twarm', type=int, default=10, 
                        help='Warm-up epochs for Phase 1')
    
    # 4. 权重平衡因子 (对应公式中的 lambda)
    parser.add_argument('--weight_balance', type=float, default=0.5, 
                        help='Balance factor lambda between prior consistency and feature consistency')
    
    # 5. 标记频率 c (即 label_frequency)
    parser.add_argument('--label_freq', type=float, default=0.5, 
                        help='Frequency c of labeled positives (P(s=1|y=1))')



def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--csv_log', action='store_true',
                        # default=True,
                        help='Enable csv logging')
