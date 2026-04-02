import pandas as pd

def normal_paths(df: pd.DataFrame) -> dict:
    """
    데이터프레임을 {TRIP_NO: trajectory} 형태의 딕셔너리로 변환
    trajectory는 시간순으로 정렬된 [DYNA_DYN_KD_CD, DPR_MT1_UNIT_TM, DPR_CELL_XCRD, DPR_CELL_YCRD]의 numpy 배열

    Args:
        df (pd.DataFrame): 정제된 데이터프레임

    Returns:
        dict: {TRIP_NO: trajectory} 형태의 딕셔너리
    """
    trajectories = {}
    df = df.copy()
    df['DPR_MT1_UNIT_TM'] = pd.to_datetime(df['DPR_MT1_UNIT_TM'])
    for no, group in df.groupby('TRIP_NO'):
        group.sort_values('DPR_MT1_UNIT_TM', inplace=True)
        trajectories[no] = group[['DYNA_DYN_KD_CD', 'DPR_MT1_UNIT_TM', 'DPR_CELL_XCRD', 'DPR_CELL_YCRD']].to_numpy()
    
    return trajectories

def transport_path(df: pd.DataFrame) -> dict:
    """
    normal_paths와 같은 구조에 TRANSPORT_TYPE 열이 추가된 형태로 변환
    공항버스 이전의 경로를 구하기 위해 TRANSPORT_TYPE이 None이 아닌 첫 번째 지점 이전의 지점들로만 구성
    
    Args:
        df (pd.DataFrame): 공항버스가 매핑된 데이터프레임
        
    Returns:
        dict: {TRIP_NO: trajectory} 형태의 딕셔너리
    """
    trajectories = {}
    df = df.copy()
    df['DPR_MT1_UNIT_TM'] = pd.to_datetime(df['DPR_MT1_UNIT_TM'])
    df['TRANSPORT_TYPE'] = df['TRANSPORT_TYPE'].apply(lambda x: None if pd.isna(x) else x)
    for no, group in df.groupby('TRIP_NO'):
        group.sort_values('DPR_MT1_UNIT_TM', inplace=True)
        trajectories[no] = group[['DYNA_DYN_KD_CD', 'DPR_MT1_UNIT_TM', 'DPR_CELL_XCRD', 'DPR_CELL_YCRD', 'TRANSPORT_TYPE']].to_numpy()
    
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