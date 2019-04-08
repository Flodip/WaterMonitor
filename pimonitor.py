from sense_hat import SenseHat
import psycopg2
import numpy as np
import time

sense = SenseHat()
sense.set_imu_config(True, False, False)  # compass, not gyro, not accel

database = "watermonitor"

try:
    try:
        conn = psycopg2.connect(
            user="pi",
            password="piwater",
            host="127.0.0.1",
            port="5432",
            database=database
        )
    except Exception:
        message = "Error db conn"
        raise

    while True:
        # time.sleep(0.02) # already a lag of 0.02s without sleep
        xyz = sense.get_compass_raw()  # get values in microteslas

        # get timestamp in ms
        timestamp = int(round(time.time() * 1000))
        # get norm of compass xyz values
        value = np.linalg.norm([xyz["x"], xyz["y"], xyz["z"]])
        try:
            curs = conn.cursor()
            print(str(timestamp) + ", " + str(value))
            curs.execute("INSERT INTO water_consumption (timestamp, value) VALUES(%s, %s);", (timestamp, value))
            conn.commit()
            curs.close()
        except Exception:
            message = "Error cursor db"
            raise
except Exception as e:
    print(message + str(e))
