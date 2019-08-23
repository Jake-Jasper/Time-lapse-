from time import sleep
from picamera import PiCamera
import os

save_location = "/media/pi/50E6-BCEC/Test location"
os.chdir(save_location)
camera = PiCamera(resolution=(1920,1080)) # max resolution
# Set ISO to the desired value
camera.iso = 100
camera.rotation =180
# Wait for the automatic gain control to settle
sleep(5)
# Now fix the values
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'
g = camera.awb_gains
camera.awb_mode = 'off'
camera.awb_gains = g


# instead of count I could have a storage sense thing
count = 0
while count < 100:

    WAIT_TIME = 60
    for filename in camera.capture_continuous('img{timestamp:%Y-%m-%d-%H-%M-%S}.png',format = 'png', quality = 100):
                sleep(WAIT_TIME)
                count += 1

