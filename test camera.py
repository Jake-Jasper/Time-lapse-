import picamera

# from picamera import PiCamera as camera
import os
import time
import testing_lights as lights
from time import sleep
import datetime

save_location = "/media/pi/50E6-BCEC/Test location"
os.chdir(save_location)
# camera = PiCamera(PiCamera.MAX_RESOLUTION) # max resolution


def time_lapse():

    lights.lights_on()
    camera = picamera.PiCamera()
    camera.resolution = (3280, 2464)
    # Set ISO to the desired value
    camera.iso = 400
    camera.brightness = 55
    # Wait for the automatic gain control to settle
    sleep(5)
    # Now fix the values
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = "off"
    g = camera.awb_gains
    camera.awb_mode = "off"
    camera.awb_gains = g

    time.sleep(5)

    name = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")

    output = "{}.jpg".format(name)

    print(output)
    camera.capture(output=output, quality=100)

    lights.lights_off()
    camera.close()

    time.sleep(3600)


while True:
    time_lapse()
