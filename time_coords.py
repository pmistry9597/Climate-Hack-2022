from datetime import datetime

def str2datetime(strVal: str):
    return datetime.strptime(strVal, '%Y-%m-%d %H:%M:%S')