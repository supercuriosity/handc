
AUDIO_DEVICE_ID = 2

ANGLE_SENSOR = {
    "smbus_id": 2,
    "i2c_address": 0x36
}

FORCE_SENSOR = {
    "left": {"smbus_id": 3,
             "i2c_address": 0x48,},
    "right": {"smbus_id": 4,
              "i2c_address": 0x48,}
}

TACTILE_CAMERA = {
    "left":"/dev/video2",
    "right":"/dev/video0"
}

WRIST_CAMERA = {
    "wrist":"/dev/video4"
}