import sys
import asyncio
from sense_hat import SenseHat
import time

async def waitforthat(future):
	# await asyncio.sleep(0.02)
	future.set_result("ok")

async def getcompass():
	print(sense.get_compass_raw())

sense = SenseHat()
sense.set_imu_config(True, False, False) #only compass active

t = int(round(time.time() * 1000))
while True:
	loop = asyncio.get_event_loop()
	future = asyncio.Future()
	asyncio.ensure_future(waitforthat(future))
	t_prev = t
	t = int(round(time.time() * 1000))
	loop.run_until_complete(getcompass())
	loop.run_until_complete(future)
	print(t - t_prev)
loop.close()
