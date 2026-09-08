"""新数据从头训练 -> 设置当前预测模型 -> 输出训练图。"""
from nlms_dfe.common import ROOT,read_config,write_json
from nlms_dfe.training.engine import train


def main():
    config=read_config('configs/train.json')
    if config['training']['mode']!='all_data' or config['data']['fields']['ber']!='mean_coded_ber':
        raise ValueError('此入口要求全量训练且目标字段为mean_coded_ber。')
    checkpoint=train(config)  # 创建随机初始化网络；没有加载旧权重。
    try:
        checkpoint_name=str(checkpoint.relative_to(ROOT))
    except ValueError:
        checkpoint_name=str(checkpoint)
    write_json(ROOT/'configs/inference.json',{'checkpoint':checkpoint_name,'target_metric':'MeanCodedBER',
               'label':f"MeanCodedBER · {config['model']['layers']}层时间网络"})
    from scripts.plot_training import plot
    plot(checkpoint.parent)
    print('训练和出图完成。现在可以打开预测窗口。')


if __name__=='__main__':
    main()
