"""新信道+SNR -> 参数JSON与MAT，供用户的MATLAB接收机比较。"""
import argparse
import json
from scipy.io import savemat

from nlms_dfe.common import project_path, write_json, active_model
from nlms_dfe.data.io import read_field
from nlms_dfe.predict import Predictor


def main():
    parser = argparse.ArgumentParser(description='输入新信道及SNR，导出均衡器参数')
    parser.add_argument('--channel', required=True)
    parser.add_argument('--snr-db', required=True, type=float)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--field', default='h_vary')
    parser.add_argument('--main-path', type=int, default=None, help='原始主径MATLAB列号，默认626；新数据可设501')
    parser.add_argument('--already-cropped', action='store_true')
    parser.add_argument('--allow-short', action='store_true', help='不足4000行时允许使用真实末尾')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='outputs/new_channel_prediction.json')
    args = parser.parse_args()
    checkpoint = args.checkpoint or active_model()['checkpoint']
    predictor = Predictor(project_path(checkpoint), args.device)
    h = read_field(project_path(args.channel), args.field,
                   predictor.checkpoint['config']['data']['hdf5_matlab_order'])
    result = predictor.predict(h, args.snr_db, already_cropped=args.already_cropped,
                               main_path_matlab=args.main_path, allow_short=args.allow_short)
    result.update(channel_path=str(project_path(args.channel)), snr_db=args.snr_db,
                  input_already_cropped=args.already_cropped,
                  main_path_matlab=args.main_path if args.main_path is not None else predictor.checkpoint['config']['data']['crop']['main_path_matlab'])
    output = project_path(args.output)
    write_json(output, result)
    # MATLAB直接load即可得到mu/N1/N2，不需要解析JSON。
    selected = result['selected']
    savemat(output.with_suffix('.mat'), {k:selected[k] for k in ('mu','N1','N2','candidate_id')})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
