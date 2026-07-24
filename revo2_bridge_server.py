import asyncio
import json
import websockets
import sys
import traceback

try:
    from bc_stark_sdk import main_mod as sdk
except ImportError:
    try:
        import libstark as sdk
    except ImportError:
        print("Error: bc_stark_sdk or libstark not found.")
        print("Please install SDK first: pip install bc_stark_sdk")
        sys.exit(1)

import argparse

# 存储当前连接的机械手设备上下文和 ID
revo2_device = None
ARGS = None

async def interactive_setup():
    global ARGS
    
    # 获取可用端口
    try:
        ports_bytes = sdk.list_available_ports()
        ports_list = json.loads(ports_bytes.decode("utf-8"))
    except Exception as e:
        print(f"获取可用端口失败: {e}")
        ports_list = []
        
    try:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            if not any(x['port_name'] == p.device for x in ports_list):
                ports_list.append({'port_name': p.device, 'description': p.description})
    except ImportError:
        pass

    selected_port = ARGS.port if ARGS and ARGS.port else None
    
    if not selected_port:
        if not ports_list:
            print("未检测到任何可用串口，请手动输入串口名称 (例如 COM3 或 /dev/ttyUSB0):")
            selected_port = input(">>> ").strip()
        else:
            print("\n检测到以下可用串口:")
            for i, p in enumerate(ports_list):
                print(f"[{i + 1}] {p['port_name']}")
            print("[0] 手动输入其他串口")
            
            while True:
                choice = input("\n请选择串口编号 (默认=1): ").strip()
                if not choice:
                    selected_port = ports_list[0]['port_name']
                    break
                try:
                    idx = int(choice)
                    if idx == 0:
                        selected_port = input("请输入串口名称: ").strip()
                        break
                    elif 1 <= idx <= len(ports_list):
                        selected_port = ports_list[idx - 1]['port_name']
                        break
                    else:
                        print("无效的选择，请重试")
                except ValueError:
                    print("请输入数字")

    slave_id = ARGS.id if ARGS and ARGS.id else None
    if not slave_id:
        id_str = input(f"\n请输入设备 ID (右=127, 左=126, 默认=127): ").strip()
        if not id_str:
            slave_id = 127
        else:
            try:
                slave_id = int(id_str)
            except ValueError:
                print("无效输入，将使用默认 ID 127")
                slave_id = 127
                
    baudrate = ARGS.baud if ARGS and ARGS.baud else 460800

    return selected_port, baudrate, slave_id

async def init_revo2():
    global revo2_device
    print("\n正在配置连接（支持 Revo2 RS-485/Modbus）...")
    try:
        port_name, baud, slave_id = await interactive_setup()
        
        baud_map = {
            115200: sdk.Baudrate.Baud115200,
            460800: sdk.Baudrate.Baud460800,
            1000000: sdk.Baudrate.Baud1Mbps,
            2000000: sdk.Baudrate.Baud2Mbps,
            5000000: sdk.Baudrate.Baud5Mbps,
        }
        baudrate = baud_map.get(baud, sdk.Baudrate.Baud460800)
        
        print(f"\n尝试连接: 端口={port_name}, 波特率={baud}, ID={slave_id}")
        
        ctx = await sdk.modbus_open(port_name, baudrate)
        
        # 对于 Revo2，进行初始化设定
        await ctx.set_hardware_type(slave_id, sdk.StarkHardwareType.Revo2Basic)
        await ctx.set_finger_unit_mode(slave_id, sdk.FingerUnitMode.Normalized)
        
        try:
            device_info = await ctx.get_device_info(slave_id)
            print(f"设备信息: {device_info.description}")
        except Exception as e:
            print("获取设备信息警告:", e)

        print("机械手连接成功！")
        revo2_device = {
            "ctx": ctx,
            "slave_id": slave_id
        }
        return True
    except Exception as e:
        print(f"初始化 Revo2 失败: {e}")
        traceback.print_exc()
        return False


async def handle_client(websocket):
    global revo2_device
    print("网页前端已连接到本地 Python 网桥")
    if not revo2_device:
        print("注意: 当前没有连接到机械手设备。")
        
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "positions":
                    # 期望接收数组: [Thumb, ThumbAux, Index, Middle, Ring, Pinky] (0~1000)
                    positions = data.get("data", [0]*6)
                    if revo2_device:
                        # 使用 30ms 作为预期持续时间，与前端 33Hz 的更新频率匹配，实现平滑插值，减少抖动
                        durations = [30] * 6 
                        
                        async def send_cmd(slave_id, pos, dur):
                            try:
                                await revo2_device["ctx"].set_finger_positions_and_durations(slave_id, pos, dur)
                            except Exception as e:
                                # 忽略密集发送期间可能出现的冲突错误
                                pass
                                
                        # 使用 create_task 避免阻塞 WebSocket 接收循环，从而消除延迟积压
                        asyncio.create_task(send_cmd(revo2_device["slave_id"], positions, durations))
            except Exception as e:
                print(f"处理指令或调用 SDK 失败: {e}")
    except websockets.exceptions.ConnectionClosed:
        print("网页前端连接已断开")

async def main():
    global ARGS
    parser = argparse.ArgumentParser(description="脑机 Revo2 本地桥接服务")
    parser.add_argument("-p", "--port", type=str, help="指定串口，例如 COM3 或者 /dev/ttyUSB0", default=None)
    parser.add_argument("-b", "--baud", type=int, help="指定波特率，默认 460800", default=460800)
    parser.add_argument("-i", "--id", type=int, help="指定设备ID，默认右手127", default=127)
    ARGS = parser.parse_args()

    print("=== BrainCo SDK 本地桥接服务 ===")
    connected = await init_revo2()
    if not connected:
        print("\n注意：由于未连接到机械手，目前只启动 WebSocket 服务用于测试通讯。")
        print("如果您已经确保硬件插好，稍后前端连接时会自动重试。\n")

    # 启动 WebSocket 服务器供网页端连接
    # 在 0.0.0.0 上监听，可支持跨越本机IP连接（如果有需要）
    start_server = websockets.serve(handle_client, "0.0.0.0", 8765)
    print("本地桥接服务已启动，监听 WebSocket 于 ws://localhost:8765")
    print("请回到网页端，点击【通过官方 SDK 桥接控制】按钮")
    
    await start_server
    await asyncio.Future()  # 保持运行

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n退出桥接服务")
        sys.exit(0)
    except Exception as e:
        print(f"服务异常退出: {e}")
        sys.exit(1)
