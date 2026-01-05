import pandas as pd

def update_air_bus_output(output, airport_bus):
    # 공항버스 TRIP_NO별 데이터 업데이트
    for trip_no, values in airport_bus.items():
        for value in values:
            mask = (output['TRIP_NO'] == trip_no) & (pd.to_datetime(output['DPR_MT1_UNIT_TM']) == value[0])
            if value[4] == '공항버스':
                output.loc[mask, 'BUS_ID'] = ', '.join(map(str, value[1]))
                output.loc[mask, 'STATION'] = ', '.join(map(str, value[3]))
                output.loc[mask, 'TRANSPORT_TYPE'] = value[4]
    
    return output

def update_city_bus_output(output, city_bus):
    # 도시버스 TRIP_NO별 데이터 업데이트
    for trip_no, values in city_bus.items():
        for value in values:
            mask = (output['TRIP_NO'] == trip_no) & (pd.to_datetime(output['DPR_MT1_UNIT_TM']) == value[0])
            if value[3] == '일반버스':
                output.loc[mask, 'BUS_ID'] = ', '.join(map(str, value[1]))
                output.loc[mask, 'STATION'] = ', '.join(map(str, value[2]))
                output.loc[mask, 'TRANSPORT_TYPE'] = value[3]
                
    return output