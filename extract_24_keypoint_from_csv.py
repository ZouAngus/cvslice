import pandas as pd
from tqdm import tqdm
import numpy as np


TARGET_JOINTS_ORDERED = {
    0:  [('Hip','Bone')],
    1:  [('LThigh','Bone')],
    2:  [('RThigh','Bone')],
    3:  [('Ab','Bone')],
    4:  [('LShin','Bone')],
    5:  [('RShin','Bone')],
    6:  [('BackLeft','Bone Marker'),('BackRight','Bone Marker')],
    7:  [('LFoot','Bone')],
    8:  [('RFoot','Bone')],
    9:  [('BackTop','Bone Marker')],
    10: [('LToe','Bone')],
    11: [('RToe','Bone')],
    12: [('Neck','Bone')],
    13: [('LShoulder','Bone')],
    14: [('RShoulder','Bone')],
    15: [('Head','Bone')],
    16: [('LUArm','Bone')],
    17: [('RUArm','Bone')],
    18: [('LFArm','Bone')],
    19: [('RFArm','Bone')],
    20: [('LWristIn','Bone Marker'),('LWristOut','Bone Marker')],
    21: [('RWristIn','Bone Marker'),('RWristOut','Bone Marker')],
    22: [('LHandOut','Bone Marker')],
    23: [('RHandOut','Bone Marker')],
}


def extract_3d_points_from_csv(input_path: str, output_path: str, total_frames: int = -1,skiprows: int = 1, offset: int = 0):
    df = pd.read_csv(input_path, skiprows=skiprows, low_memory=False)
    type_list = list(df.iloc[0].index)
    header_row = df.iloc[0]
    if offset < 0:
        # raise ValueError("Offset must be a non-negative integer.")
        print("Warning: Offset must be a non-negative integer.")
        start = 4 + offset
    else:
        start = 4

    if total_frames > 0:  # has been set
        end = start + total_frames
        if end > len(df):
           print("Total frames exceed the number of available frames in the CSV file.")
    else:
        end = len(df)

    df = df.iloc[start:end].reset_index(drop=True)

    selected_columns = []

    for joint_id in range(24):
        joint_defs = TARGET_JOINTS_ORDERED[joint_id]
        if len(joint_defs) == 2:
            cols_0 = [i for i, val in enumerate(header_row)
                      if str(val)[13:] == joint_defs[0][0] and type_list[i].startswith(joint_defs[0][1])][-3:]
            cols_1 = [i for i, val in enumerate(header_row)
                      if str(val)[13:] == joint_defs[1][0] and type_list[i].startswith(joint_defs[1][1])][-3:]
            selected_columns.append((cols_0, cols_1))
        else:
            cols = [i for i, val in enumerate(header_row)
                    if str(val)[13:] == joint_defs[0][0] and type_list[i].startswith(joint_defs[0][1])][-3:]
            selected_columns.append(cols)

    final_data = []
    for frame_idx, row in tqdm(df.iterrows(), total=len(df), desc="提取3D点"):
        frame_data = []
        try:
            for cols in selected_columns:
                if isinstance(cols, tuple):  # 平均两个 marker
                    vals_0 = pd.to_numeric(row.iloc[cols[0]], errors='coerce').values
                    vals_1 = pd.to_numeric(row.iloc[cols[1]], errors='coerce').values
                    avg = (vals_0 + vals_1) / 2
                    frame_data.extend(avg)
                else:
                    vals = pd.to_numeric(row.iloc[cols], errors='coerce').values
                    frame_data.extend(vals)
            final_data.append(frame_data)
        except Exception as e:
            print(f"\n❌ 第 {frame_idx} 帧出错：{e}")
            # 可选：跳过这一帧，或填入 NaN
            frame_data = [np.nan] * (24 * 3)
            final_data.append(frame_data)

    if offset < 0:  # 补充视频前面几帧 丢失的数据
        for i in range(abs(offset)):
            final_data = [[np.nan] * (24 * 3)] + final_data
    columns = [f"{i}_{axis}" for i in range(24) for axis in ['x', 'y', 'z']]
    df_out = pd.DataFrame(final_data, columns=columns)
    df_out.to_csv(output_path, index=False)
    print(f"\n 提取完成，结果已保存至 {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract 24-joint 3D keypoints from an OptiTrack CSV file.")
    parser.add_argument("-input_csv",   required=False, default="./Trove_15.csv",            help="Path to the input OptiTrack CSV file")
    parser.add_argument("-output_csv",  required=False, default="./trove_15_3d_points_out.csv", help="Path to save the output CSV file")
    parser.add_argument("-total_frames",required=False, default=-1,   type=int, help="Number of frames to extract (-1 = all)")
    parser.add_argument("-skiprows",    required=False, default=1,    type=int, help="Number of header rows to skip (default: 1)")
    parser.add_argument("-offset",      required=False, default=4,    type=int, help="Frame offset into the CSV (default: 4)")
    args = parser.parse_args()

    extract_3d_points_from_csv(
        input_path=args.input_csv,
        output_path=args.output_csv,
        total_frames=args.total_frames,
        skiprows=args.skiprows,
        offset=args.offset,
    )


