import pandas as pd

def update_air_bus_output(output: pd.DataFrame, airport_bus: dict) -> pd.DataFrame:
    """
    추출된 공항버스 매핑 데이터와 원본 데이터 병합
    value: [time, bus_id, ars_id, station_name, bus_type]

    Args:
        output (pd.DataFrame): 원본 데이터
        airport_bus (dict): 공항버스 매핑 데이터

    Returns:
        pd.DataFrame: value 값이 추가된 output
    """
    for trip_no, values in airport_bus.items():
        for value in values:
            mask = (output['TRIP_NO'] == trip_no) & (pd.to_datetime(output['DPR_MT1_UNIT_TM']) == value[0])
            if value[4] == '공항버스':
                output.loc[mask, 'BUS_ID'] = ', '.join(map(str, value[1]))
                output.loc[mask, 'STATION'] = ', '.join(map(str, value[3]))
                output.loc[mask, 'TRANSPORT_TYPE'] = value[4]
    
    return output

def update_city_bus_output(output: pd.DataFrame, city_bus: dict) -> pd.DataFrame:
    """
    추출된 도시버스 매핑 데이터와 공항버스가 매핑된 원본 데이터의 병합
    value: [time, bus_id, station_name, bus_type]

    Args:
        output (pd.DataFrame): 공항버스가 매핑된 원본 데이터
        city_bus (dict): 도시버스 매핑 데이터

    Returns:
        pd.DataFrame: value 값이 추가된 output
    """
    for trip_no, values in city_bus.items():
        for value in values:
            mask = (output['TRIP_NO'] == trip_no) & (pd.to_datetime(output['DPR_MT1_UNIT_TM']) == value[0])
            if value[3] == '일반버스':
                output.loc[mask, 'BUS_ID'] = ', '.join(map(str, value[1]))
                output.loc[mask, 'STATION'] = ', '.join(map(str, value[2]))
                output.loc[mask, 'TRANSPORT_TYPE'] = value[3]
                
    return output