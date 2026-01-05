import pandas as pd

# 시간을 datetime 형식으로 변환
def time_to_datatime(df):
    df['DPR_MT1_UNIT_TM'] = pd.to_datetime(df['DPR_MT1_UNIT_TM'])
    return df

# TRANSPORT_TYPE의 값이 nan이면 None으로 변경
def transport_nan_to_none(df):
    df['TRANSPORT_TYPE'] = df['TRANSPORT_TYPE'].apply(lambda x: None if pd.isna(x) else x)
    return df

# 각 사람별로 데이터 그룹화
def group_py_NO(df):
    grouped = df.groupby('TRIP_NO')
    return grouped

# 기본 이동 경로 생성 함수
def normal_create_trajectory(group):
    return group.sort_values('DPR_MT1_UNIT_TM')[['DYNA_DYN_KD_CD', 'DPR_MT1_UNIT_TM', 'DPR_CELL_XCRD', 'DPR_CELL_YCRD']].to_numpy()

# transport 이동 경로 생성 함수
def transport_create_trajectory(group):
    return group.sort_values('DPR_MT1_UNIT_TM')[['DYNA_DYN_KD_CD', 'DPR_MT1_UNIT_TM', 'DPR_CELL_XCRD', 'DPR_CELL_YCRD', 'TRANSPORT_TYPE']].to_numpy()

# 각 사람별 기본 이동 경로 생성
def normal_paths(df):
    df = time_to_datatime(df)
    grouped = group_py_NO(df)
    trajectories = {no: normal_create_trajectory(group) for no, group in grouped}
    return trajectories

def transport_path(df):
    df = time_to_datatime(df)
    df = transport_nan_to_none(df)
    grouped = group_py_NO(df)
    trajectories = {no: transport_create_trajectory(group) for no, group in grouped}
    
    peopletraj = {}
    for key, values in trajectories.items():
        temp = []
        for value in values:
            if value[4] is not None:
                break
            temp.append(value)
        if temp:
            peopletraj[key] = temp
    return peopletraj