import rclpy

print("STEP 1: rclpy.init()")
rclpy.init()
print("STEP 1 OK")

print("STEP 2: rclpy.create_node()")
node = rclpy.create_node("g1_voice_test")
print("STEP 2 OK")

print("STEP 3: Import Unitree SDK")
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
print("STEP 3 OK")

print("STEP 4: Unitree ChannelFactoryInitialize(0, 'eth0')")
ChannelFactoryInitialize(0, "eth0")
print("STEP 4 OK")

print("STEP 5: AudioClient")
client = AudioClient()
client.SetTimeout(10.0)
client.Init()
print("STEP 5 OK")

print("STEP 6: TTS")
result = client.TtsMaker(
    "This is the G1 ROS node integration test.",
    0
)
print("STEP 6 TTS return code:", result)

node.destroy_node()
rclpy.shutdown()

print("DONE")
