import rclpy

print("STEP 1: Initializing ROS 2...")
rclpy.init()
print("STEP 1: ROS 2 initialized OK")

print("STEP 2: Importing Unitree SDK...")
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
print("STEP 2: Unitree SDK imported OK")

print("STEP 3: Initializing Unitree DDS on eth0...")
ChannelFactoryInitialize(0, "eth0")
print("STEP 3: Unitree DDS initialized OK")

print("STEP 4: Initializing G1 AudioClient...")
client = AudioClient()
client.SetTimeout(10.0)
client.Init()
print("STEP 4: G1 AudioClient initialized OK")

print("STEP 5: Sending TTS...")
result = client.TtsMaker(
    "This is a ROS and Unitree DDS integration test.",
    0
)
print("STEP 5: TTS return code:", result)

rclpy.shutdown()
print("DONE")
