from __future__ import annotations

import asyncio
import collections
import json
import math
import os
import re
import subprocess
import threading
import time
import wave
from pathlib import Path

import dashscope
import edge_tts
import numpy as np
import pyaudio
import rclpy
from dashscope.audio.asr import Recognition
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String

from action_msgs.msg import GoalStatusArray, GoalStatus


#音频与路径配置
BASE_DIR = Path(__file__).parent
TEMPLATE_AUDIO_DIR = BASE_DIR / "template_audio"
TEMPLATE_MANIFEST = TEMPLATE_AUDIO_DIR / "manifest.json"

SILENCE_FILE = "/tmp/silence_warmup.mp3"
WAKEUP_FILE = "/home/unitree/audio/wakeup_prompt.mp3"
TTS_FALLBACK_FILE = "/tmp/tts_reply.mp3"


#播放设备配置
def _find_speaker_device() -> str:
    """从 `aplay -l` 中查找 USB 音响，并返回 ALSA 设备字符串。"""
    try:
        out = subprocess.check_output(["aplay", "-l"], stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if line.startswith("card ") and "USB2.0 Device" in line:
                card_num = line.split(":")[0].replace("card ", "").strip()
                return f"alsa/hw:{card_num},0"
    except Exception:
        pass
    return "alsa/hw:0,0"


SPEAKER_DEVICE = _find_speaker_device()


def _ensure_silence_file() -> None:
    """生成一段短静音文件，用于唤醒扬声器硬件。"""
    if not os.path.exists(SILENCE_FILE):
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                "-t", "0.5", "-q:a", "9", SILENCE_FILE,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


_ensure_silence_file()


#业务与音频参数
ALIYUN_API_KEY = "************************"  #填入自己的API密钥
dashscope.api_key = ALIYUN_API_KEY


# 新增这一行，填入刚才生成的真实 ID   (热词专用)
#HOTEL_VOCABULARY_ID = "wlh_xxxxxxxxx" # 替换为create_vocab.py实际获取的 ID

MIC_DEVICE_HINTS = ["HK-MIC", "HKMIC"]
MIC_RATE = 48000
MIC_CHANNELS = 2
MIC_INPUT_INDEX = None
MIC_USE_AUTO_DOWNMIX = True
TARGET_MIC_CHANNEL = 0
MIC_GAIN_PER_CHANNEL = 1.0

KWS_RATE = 16000
KWS_CHUNK = 9600
REC_CHUNK = 4800
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
WAVE_OUTPUT_FILENAME = "/tmp/user_voice.wav"
SOFTWARE_GAIN = 2.0

JSON_PATH = "/home/unitree/ai_audio/goal_json/hotel_waypoints.json"

VOLUME_THRESHOLD = 0.015
SILENCE_LIMIT_SEC = 1.2
MAX_RECORD_SEC = 10
MIN_RECORD_SEC = 1.5
WAKE_KEYWORD = "你好小万"
WAKE_KEYWORDS = ["你好小万", "你好小王", "你好小玩", "你好小丸", "你好小汪", "你好小"]
WAKE_ASR_MIN_CHARS = 4
WAKE_CONFIRM_REPEAT = 2
WAKE_LISTEN_MAX_SEC = 8
WAKE_MIN_AUDIO_SEC = 1.8
WAKE_CHECK_INTERVAL_SEC = 1.0

_FIR_CUTOFF = 8000.0 / 48000.0
_FIR_TAPS = 51
_n = np.arange(_FIR_TAPS) - (_FIR_TAPS - 1) / 2
_h = np.sinc(2 * _FIR_CUTOFF * _n) * np.hamming(_FIR_TAPS)
_h /= _h.sum()



def resample_48k_to_16k(buf: np.ndarray) -> np.ndarray:
    """将 48kHz 音频低通后抽取为 16kHz。"""
    return np.convolve(buf, _h, mode='same')[::3].astype(np.float32)



SYSTEM_PROMPT = """
你是西安万丽酒店的智能服务机器人。你必须理解用户意图并输出纯 JSON 格式。
格式要求：{"action": "navigate"或"speak", "target": "地点名称或none", "content": "你要说的话"}
注意：绝不输出 markdown 或额外解释文字。


【语音纠错指南】（非常重要）：
由于语音识别(ASR)可能存在误差，请结合酒店场景进行模糊纠错。例如：
- 客人说“去里边跑”、“李冰台”、“里宾台”等发音相似的词，实际上是指“礼宾台”。
- 客人说“前排”、“前台”等，指“前台”。
- 客人说“私餐厅”、“四餐厅”、“诗餐厅”、“吃餐厅”、“去13天”等餐厅前有与“丝”发音相似的词，实际上是指“丝餐厅”。
请直接将纠错后的正确地点填入 target 中。

【内部地点导航列表】（仅能从中选取）：
前台, 电梯, 燃餐厅, 万丽轩中餐厅, 健身房, 泳池, 丝餐厅, 卫生间, 大堂吧台,礼宾台

【严格执行以下语音包模板】：
- 迎宾问候：{"action": "speak", "target": "none", "content": "您好，有什么可以帮您？我可以带路、送物、解答问题。"}
- 引导前台：{"action": "navigate", "target": "前台", "content": "您好，办理入住或退房，请随我前往前台。办理入住需要出示有效身份证件。"}
- 引导电梯：{"action": "navigate", "target": "电梯", "content": "电梯在这边，请跟我来。"}
- 引导燃餐厅：{"action": "navigate", "target": "燃餐厅", "content": "燃餐厅在酒店一楼，我带您前往。"}
- 引导中餐厅：{"action": "speak", "target": "万丽轩中餐厅", "content": "万丽轩中餐厅在6楼"}
- 引导健身房/泳池：{"action": "speak", "target": "健身房", "content": "健身房与泳池位于5楼，泳池营业时间6:00-23:00。健身房24小时营业。"}
- 引导丝餐厅：{"action": "navigate", "target": "丝餐厅", "content": "丝餐厅位于酒店一楼，午餐营业时间上午11:30-13:30；晚餐营业时间17:30-21:00。"}
- 引导卫生间：{"action": "navigate", "target": "卫生间", "content": "洗手间在前方，请跟我来。"}
- 引导大堂吧台：{"action": "navigate", "target": "大堂吧台", "content": "好的，我这就带您前往大堂吧台。"}
- 引导礼宾台：{"action": "navigate", "target": "礼宾台", "content": "好的，礼宾台在这边，请跟我来。"}
- 询问早餐时间：{"action": "speak", "target": "none", "content": "早餐在1楼燃西餐厅，6:30-10:00。"}
- 询问WiFi：{"action": "speak", "target": "none", "content": "酒店Wi-Fi：万豪旅享家，输入您的房号和您的姓氏拼音就可以啦。"}
- 询问停车：{"action": "speak", "target": "none", "content": "酒店提供地下停车场，住店客人可前台登记免停车费。"}
- 询问周边景点：{"action": "speak", "target": "none", "content": "酒店临近大雁塔、大唐不夜城、大唐芙蓉园，出行便利。"}
- 客人道谢/夸赞：{"action": "speak", "target": "none", "content": "感谢您的认可，期待与您再次相遇。很乐意为您服务。"}
- 遇到不知道的问题：{"action": "speak", "target": "none", "content": "抱歉，我暂时无法解答，请您联系工作人员。"}
"""

# 预制音频键映射：LLM 返回的 content 命中这些模板时，优先播放本地 mp3
TEMPLATE_AUDIO_MAP = {
    "您好，有什么可以帮您？我可以带路、送物、解答问题。": "greet.mp3",
    "您好，办理入住或退房，请随我前往前台。办理入住需要出示有效身份证件。": "front_desk.mp3",
    "电梯在这边，请跟我来。": "elevator.mp3",
    "燃餐厅在酒店一楼，我带您前往。": "burn_restaurant.mp3",
    "万丽轩中餐厅在6楼": "wanli_xuan.mp3",
    "健身房与泳池位于5楼，泳池营业时间6:00-23:00。健身房24小时营业。": "gym_pool.mp3",
    "丝餐厅位于酒店一楼，午餐营业时间上午11:30-13:30；晚餐营业时间17:30-21:00。": "silk_restaurant.mp3",
    "洗手间在前方，请跟我来。": "toilet.mp3",
    "早餐在1楼燃西餐厅，6:30-10:00。": "breakfast.mp3",
    "酒店Wi-Fi：万豪旅享家，输入您的房号和您的姓氏拼音就可以啦。": "wifi.mp3",
    "酒店提供地下停车场，住店客人可前台登记免停车费。": "parking.mp3",
    "酒店临近大雁塔、大唐不夜城、大唐芙蓉园，出行便利。": "sightseeing.mp3",
    "感谢您的认可，期待与您再次相遇。很乐意为您服务。": "thanks.mp3",
    "抱歉，我暂时无法解答，请您联系工作人员。": "unknown.mp3",
    "感谢您的使用，祝您旅途愉快，再见！": "goodbye.mp3",
    "好的，我这就带您前往大堂吧台。": "lobby_bar.mp3",
    "好的，我这就带您前往大门。": "door.mp3",
    "好的，礼宾台在这边，请跟我来。": "concierge.mp3",
    "已到达目的地，前台到了，祝您办理顺利。": "arrived_front_desk.mp3",
    "已到达目的地，电梯到了，请您乘坐。": "arrived_elevator.mp3",
    "已到达目的地，燃餐厅到了，请慢用。": "arrived_burn_restaurant.mp3",
    "已到达目的地，万丽轩中餐厅到了，请慢用。": "arrived_wanli_xuan.mp3",
    "已到达目的地，健身房到了，请尽情使用。": "arrived_gym.mp3",
    "已到达目的地，泳池到了，请注意安全。": "arrived_pool.mp3",
    "已到达目的地，丝餐厅到了，请慢用。": "arrived_silk_restaurant.mp3",
    "已到达目的地，洗手间到了。": "arrived_toilet.mp3",
    "已到达目的地，大堂吧到了，请慢用。": "arrived_lobby_bar.mp3",
    "已到达目的地，大门到了，祝您出行愉快。": "arrived_door.mp3",
    "已到达目的地，礼宾台到了，请稍候。": "arrived_concierge.mp3",
    "已到达目的地，万丽轩中餐厅到了，请慢用。": "arrived_wanli_xuan.mp3",
}


def _read_manifest() -> dict:
    """读取预制音频清单，若不存在则返回空字典。"""
    if TEMPLATE_MANIFEST.exists():
        try:
            return json.loads(TEMPLATE_MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}



class VoiceBrainNode(Node):
    def __init__(self):
        super().__init__('voice_brain_node_array_local_tts')

        self.pub_nav = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.pub_stop = self.create_publisher(Bool, '/go2_stop', 10)
        self.pub_tts = self.create_publisher(String, '/feedback_words', 10)
        self._current_goal_target = None
        self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self._status_callback, 10)  #这个才是导航状态订阅
        
        #新增两个变量，防止系统狂发状态导致重复播报
        self.last_status = GoalStatus.STATUS_UNKNOWN
        self.current_goal_id = None

        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                self.location_map = json.load(f)
            self.get_logger().info(f"坐标加载成功: {list(self.location_map.keys())}")
        except Exception as e:
            self.get_logger().error(f"坐标加载失败: {e}")
            self.location_map = {}

        self.p = pyaudio.PyAudio()
        self._audio_queue = collections.deque()
        self.mic_stream = None
        self.is_listening = True
        self._kws_running = True
        # 每个 chunk 是 0.2 秒 (9600 / 48000)，所以 8 秒只需要存 40 个 chunk
        self._wake_buffer = collections.deque(maxlen=int(WAKE_LISTEN_MAX_SEC * MIC_RATE / KWS_CHUNK))
        self._wake_candidate_text = ""
        self._wake_repeat_count = 0
        self._wake_last_check_time = 0.0
        self._wake_last_audio_sec = 0.0
        self._processing_lock = threading.Lock()

        self._print_input_devices()
        self._template_manifest = _read_manifest()
        threading.Thread(target=self._kws_loop, daemon=True).start()

        self._start_mic(KWS_CHUNK)
        self.get_logger().info(f"系统就绪！麦克风持续监听 ASR，检测到包含「{WAKE_KEYWORD}」的文本后唤醒。")




    #麦克风管理
    def _print_input_devices(self):
        """打印 PyAudio 输入设备列表，便于确认当前系统的录音设备。"""
        self.get_logger().info("========== PyAudio 输入设备列表 ==========")
        count = self.p.get_device_count()
        for i in range(count):
            d = self.p.get_device_info_by_index(i)
            if int(d.get('maxInputChannels', 0)) > 0:
                self.get_logger().info(
                    f"[Input] idx={i} name={d.get('name')} channels={d.get('maxInputChannels')} rate={d.get('defaultSampleRate')}"
                )
        self.get_logger().info("========================================")


    def _get_mic_index(self):
        """优先使用手工配置的输入索引，其次按设备名严格匹配麦克风。"""
        if MIC_INPUT_INDEX is not None:
            return MIC_INPUT_INDEX

        count = self.p.get_device_count()
        for i in range(count):
            d = self.p.get_device_info_by_index(i)
            name = d.get('name', '')
            if int(d.get('maxInputChannels', 0)) <= 0:
                continue
            if any(key.lower() in name.lower() for key in MIC_DEVICE_HINTS):
                self.get_logger().info(f"匹配到麦克风设备: [{i}] {name}")
                return i

        for i in range(count):
            d = self.p.get_device_info_by_index(i)
            name = d.get('name', '')
            if int(d.get('maxInputChannels', 0)) <= 0:
                continue
            low = name.lower()
            if 'ape' in low or 'default' in low or 'speaker' in low or 'output' in low:
                continue
            self.get_logger().warning(f"兜底选择输入设备: [{i}] {name}")
            return i
        return None
    
    def _downmix_to_mono(self, interleaved: np.ndarray, channels: int) -> np.ndarray:
        """将多通道 PCM 转成单声道，默认取目标通道。"""
        if channels <= 1:
            return interleaved.astype(np.float32)
        usable = (len(interleaved) // channels) * channels
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        frames = interleaved[:usable].reshape(-1, channels).astype(np.float32)
        frames *= MIC_GAIN_PER_CHANNEL
        ch = TARGET_MIC_CHANNEL
        if ch < 0 or ch >= channels:
            ch = 0
        return frames[:, ch]



    def _decode_input_bytes(self, in_data: bytes) -> np.ndarray:
        """将输入音频转换成 48kHz 单声道 float32 数据。"""
        raw = np.frombuffer(in_data, dtype=np.int16)
        mono = self._downmix_to_mono(raw, MIC_CHANNELS) if MIC_USE_AUTO_DOWNMIX else raw.astype(np.float32)
        mono = mono / 32768.0
        return np.clip(mono * SOFTWARE_GAIN, -1.0, 1.0)
    
    def _start_mic(self, chunk_size):
        """打开或重开麦克风流。"""
        if self.mic_stream is not None:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None

        idx = self._get_mic_index()
        if idx is None:
            self.get_logger().error("找不到麦克风设备！")
            return

        info = self.p.get_device_info_by_index(idx)
        max_in = int(info.get('maxInputChannels', 1))
        channels = min(MIC_CHANNELS, max_in) if max_in > 0 else 1
        self.get_logger().info(f"打开麦克风: {info.get('name')}  Index={idx}  通道={channels}")
        self._audio_queue.clear()
        self.mic_stream = self.p.open(
            format=AUDIO_FORMAT,
            channels=channels,
            rate=MIC_RATE,
            input=True,
            input_device_index=idx,
            frames_per_buffer=chunk_size,
            stream_callback=self._audio_callback,
        )
        self.mic_stream.start_stream()

    
    def _audio_callback(self, in_data, _frame_count, _time_info, _status):
        self._audio_queue.append(in_data)
        return (None, pyaudio.paContinue)
    



    #ASR 唤醒线程
    def _kws_loop(self):
        self.get_logger().info("ASR 唤醒检测线程已启动")
        while self._kws_running:
            if self._processing_lock.locked() or not self.is_listening:
                time.sleep(0.01)
                continue
            if not self._audio_queue:
                time.sleep(0.01)
                continue
            try:
                data = self._audio_queue.popleft()
                buf = self._decode_input_bytes(data)
                buf_16k = resample_48k_to_16k(buf)
                if len(buf_16k):
                    self._wake_buffer.append(buf_16k)
                self._check_wake_keyword()
            except Exception as e:
                import traceback
                self.get_logger().error(f"ASR 唤醒线程异常: {e}\n{traceback.format_exc()}")


    


    def _check_wake_keyword(self):
        if not self.is_listening or self._processing_lock.locked():
            return
        if not self._wake_buffer:
            return
        now = time.time()
        if now - self._wake_last_check_time < WAKE_CHECK_INTERVAL_SEC:
            return
        audio = np.concatenate(list(self._wake_buffer))
        audio_sec = len(audio) / float(KWS_RATE)
        if audio_sec < WAKE_MIN_AUDIO_SEC:
            return
        self._wake_last_check_time = now
        self._wake_last_audio_sec = audio_sec
        self.get_logger().info(f"唤醒检测窗口: {audio_sec:.2f}s, 样本数={len(audio)}")
        if len(audio) < int(KWS_RATE * 1.0):
            return
        try:
            temp_path = "/tmp/wake_keyword_check.wav"
            wf = wave.open(temp_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(KWS_RATE)
            wf.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
            wf.close()

            text = self._cloud_asr_file(temp_path)
            if not text:
                return
            normalized_text = re.sub(r"[\s\u3000，。！？、,.;:：!?]", "", text)
            matched_keyword = None
            for keyword in WAKE_KEYWORDS:
                normalized_keyword = re.sub(r"[\s\u3000，。！？、,.;:：!?]", "", keyword)
                if normalized_keyword in normalized_text:
                    matched_keyword = keyword
                    break
            if matched_keyword:
                self._wake_repeat_count += 1
                self._wake_candidate_text = text
                self.get_logger().info(f"ASR 监听到: {text} ({self._wake_repeat_count}/{WAKE_CONFIRM_REPEAT})")
                if self._wake_repeat_count >= WAKE_CONFIRM_REPEAT:
                    self.get_logger().info(f"唤醒词触发: 【{matched_keyword}】")
                    self.is_listening = False
                    self._wake_candidate_text = ""
                    self._wake_repeat_count = 0
                    self._wake_buffer.clear()
                    threading.Thread(target=self._execute_business, daemon=True).start()
            else:
                self._wake_candidate_text = ""
                self._wake_repeat_count = 0
        except Exception as e:
            self.get_logger().error(f"唤醒关键词检测异常: {e}")



        

    #业务主流程
    def _execute_business(self):
        with self._processing_lock:
            try:
                self.get_logger().info("已唤醒，先发送制动指令...")
                self._publish_stop_command()
                self._audio_queue.clear()
                self.get_logger().info("播放唤醒提示音...")
                if os.path.exists(WAKEUP_FILE):
                    self._play_audio_file(Path(WAKEUP_FILE))
                else:
                    self.get_logger().warning(f"唤醒提示音不存在: {WAKEUP_FILE}")
                self._record_with_vad()
                user_text = self._cloud_asr()
                if not user_text:
                    self.get_logger().info("未听清指令，恢复监听。")
                    return
                self.get_logger().info(f"识别到客人说: {user_text}")
                self._cloud_llm_and_act(user_text)
            except Exception as e:
                import traceback
                self.get_logger().error(f"业务流程异常: {e}\n{traceback.format_exc()}")
            finally:
                self._audio_queue.clear()
                self._wake_buffer.clear()
                self._wake_candidate_text = ""
                self._wake_repeat_count = 0
                self._wake_last_check_time = 0.0
                self.is_listening = True
                self.get_logger().info("处理完毕，恢复沉睡监听...")


    def _get_audio_duration(self, filepath: str) -> float:
        try:
            out = subprocess.check_output([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                filepath,
            ], stderr=subprocess.DEVNULL).decode().strip()
            return float(out)
        except Exception:
            return 1.5
        


    # VAD 录音
    def _record_with_vad(self, wakeup_start: float = None, wakeup_duration: float = 0):
        self.get_logger().info("正在倾听指令，请说话...（保留静音截断，最长录音 10 秒）")
        if self.mic_stream is not None:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None

        idx = self._get_mic_index()
        rec_stream = self.p.open(
            format=AUDIO_FORMAT,
            channels=MIC_CHANNELS,
            rate=MIC_RATE,
            input=True,
            input_device_index=idx,
            frames_per_buffer=REC_CHUNK,
        )

        if wakeup_start is not None:
            elapsed = time.time() - wakeup_start
            remain = max(0.0, wakeup_duration - elapsed) + 0.2
            discard_chunks = int(remain * MIC_RATE / REC_CHUNK)
            self.get_logger().info(f"丢弃唤醒残留音频: {discard_chunks} chunks, {remain:.2f}s")
            for _ in range(discard_chunks):
                rec_stream.read(REC_CHUNK, exception_on_overflow=False)

        max_silent_chunks = int((MIC_RATE / REC_CHUNK) * SILENCE_LIMIT_SEC)
        max_total_chunks = int((MIC_RATE / REC_CHUNK) * MAX_RECORD_SEC)
        min_total_chunks = int((MIC_RATE / REC_CHUNK) * MIN_RECORD_SEC)

        frames_16k = []
        silent_chunks = 0
        has_spoken = False

        for i in range(max_total_chunks):
            data = rec_stream.read(REC_CHUNK, exception_on_overflow=False)
            buf = self._decode_input_bytes(data)
            rms = float(np.sqrt(np.mean(buf ** 2))) if len(buf) else 0.0

            if rms > VOLUME_THRESHOLD:
                silent_chunks = 0
                if not has_spoken:
                    has_spoken = True
                    self.get_logger().info("检测到人声，开始接收指令...")
            else:
                if has_spoken:
                    silent_chunks += 1

            if has_spoken and silent_chunks > max_silent_chunks and i > min_total_chunks:
                self.get_logger().info(f"静音截断 (耗时: {round((i * REC_CHUNK) / MIC_RATE, 2)}秒)")
                break

            buf_16k = resample_48k_to_16k(buf)
            buf_16k = np.clip(buf_16k, -1.0, 1.0)
            frames_16k.append((buf_16k * 32767).astype(np.int16).tobytes())

        if not has_spoken:
            self.get_logger().info("未检测到明显声音。")

        rec_stream.close()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames_16k))
        wf.close()

        self._start_mic(KWS_CHUNK)



    #云端 ASR
    def _cloud_asr_file(self, wav_path: str) -> str:
        try:
            self.get_logger().info(f"开始ASR转文字: {wav_path}")
            recognition = Recognition(
                model='paraformer-realtime-v2',   #原来用的paraformer-realtime-v1
                format='wav',
                sample_rate=16000,
                callback=None,
            )
            res = recognition.call(wav_path)
            self.get_logger().info(f"ASR 返回状态: status_code={getattr(res, 'status_code', None)}")
            if res.status_code == 200:
                sentences = res.get_sentence()
                self.get_logger().info(f"ASR 分句结果: {sentences}")
                if sentences:
                    text = ''.join(s.get('text', '') for s in sentences).strip()
                    self.get_logger().info(f"ASR 识别文本: {text}")
                    return text
                self.get_logger().info("ASR 未返回有效文本")
            else:
                self.get_logger().error(f"ASR 失败: {res.message}")
            return ""
        except Exception as e:
            self.get_logger().error(f"云端 ASR 异常: {e}")
            return ""
        


    def _cloud_asr(self) -> str:
        return self._cloud_asr_file(WAVE_OUTPUT_FILENAME)
    


    def _publish_stop_command(self):
        msg = Bool()
        msg.data = True
        self.pub_stop.publish(msg)
        self.get_logger().info("已发布 /go2_stop 制动话题。")



    #本地音频/在线 TTS 播放 
    def _play_audio_file(self, path: Path):
        """播放本地音频文件。"""
        mpv_args = [
            "mpv",
            "--no-video",
            "--volume=130",
            "--audio-channels=stereo",
            "--no-input-default-bindings",
            "--input-terminal=no",
            "--really-quiet",
            f"--audio-device={SPEAKER_DEVICE}",
        ]
        subprocess.run(
            mpv_args + [str(path)],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )



    def _play_template_audio(self, content: str) -> bool:
        """若命中预制模板，则播放本地音频并返回 True。"""
        file_name = TEMPLATE_AUDIO_MAP.get(content)
        if not file_name:
            return False
        audio_path = TEMPLATE_AUDIO_DIR / file_name
        if not audio_path.exists():
            self.get_logger().warning(f"预制音频不存在: {audio_path}")
            return False
        self._play_audio_file(audio_path)
        return True

    def _tts_generate_and_play(self, content: str):
        """在线生成 TTS 并播放，作为预制音频的兜底方案。"""
        async def _gen():
            await edge_tts.Communicate(content, "zh-CN-XiaoyiNeural").save(TTS_FALLBACK_FILE)
        asyncio.run(_gen())
        self._play_audio_file(Path(TTS_FALLBACK_FILE))



    #云端 LLM + 执行动作
    def _cloud_llm_and_act(self, text: str):
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': text},
        ]

        try:
            self.get_logger().info("正在思考...")
            response = dashscope.Generation.call(
                model='qwen-plus',
                messages=messages,
                result_format='message',
                temperature=0.1,
            )

            if response.status_code != 200:
                self.get_logger().error(f"LLM 失败: {response.message}")
                return

            reply_text = response.output.choices[0].message.content.strip()
            self.get_logger().info(f"大模型原始返回: {reply_text}")

            clean_reply = re.sub(r'```json|```', '', reply_text).strip()
            res_json = json.loads(clean_reply)

            action = res_json.get("action", "")
            target = res_json.get("target", "none")
            content = res_json.get("content", "")

            if content:
                self.get_logger().info(f"🔊 播报: {content}")
                if not self._play_template_audio(content):
                    self._tts_generate_and_play(content)

            if action == "navigate":
                if target in self.location_map:
                    self.get_logger().info(f"下发导航目标: 【{target}】")
                    self._send_nav_goal(target, self.location_map[target])
                else:
                    self.get_logger().warning(
                        f"坐标文件中未找到地点: 【{target}】，可用地点: {list(self.location_map.keys())}"
                    )

        except json.JSONDecodeError:
            self.get_logger().error(f"大模型未输出标准 JSON: {reply_text}")
        except Exception as e:
            self.get_logger().error(f"LLM 处理异常: {e}")



    def _publish_stop_command(self):
        msg = Bool()
        msg.data = True
        self.pub_stop.publish(msg)
        self.get_logger().info("已发布 /go2_stop 制动话题。")



    def _send_nav_goal(self, target_name: str, coords: list):
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.position.x = float(coords[0])
        goal_msg.pose.position.y = float(coords[1])
        goal_msg.pose.position.z = 0.0

        yaw = float(coords[2]) if len(coords) >= 3 else 0.0
        goal_msg.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.orientation.w = math.cos(yaw / 2.0)

        self._current_goal_target = target_name
        self.pub_nav.publish(goal_msg)
        self.get_logger().info(f"已下发导航任务：前往【{target_name}】 坐标={coords}")


    def _status_callback(self, msg: GoalStatusArray):
        if not msg.status_list:
            return
            
        latest_goal_status = msg.status_list[-1]
        current_status_val = latest_goal_status.status
        current_goal_id = latest_goal_status.goal_info.goal_id.uuid
        
        # 当底层反馈状态为 4 (SUCCEEDED) 时：
        if current_status_val == GoalStatus.STATUS_SUCCEEDED:
            # 确保不是上一个任务的重复播报
            if self.last_status != GoalStatus.STATUS_SUCCEEDED or self.current_goal_id != current_goal_id:
                # 检查有没有记录的地名
                if self._current_goal_target:
                    target = self._current_goal_target
                    self._current_goal_target = None  # 取出来后立刻清空，防止下次误报
                    self.get_logger().info(f"底层标志到达！准备播报: {target}")
                    
                    # 触发写好的播报函数
                    threading.Thread(target=self._play_arrived_audio, args=(target,), daemon=True).start()
                    
        # 更新状态记录
        self.last_status = current_status_val
        self.current_goal_id = current_goal_id





    def _play_arrived_audio(self, target_name: str):
        content_map = {
            "前台": "已到达目的地，前台到了，祝您办理顺利。",
            "电梯": "已到达目的地，电梯到了，请您乘坐。",
            "燃餐厅": "已到达目的地，燃餐厅到了，请慢用。",
            "万丽轩中餐厅": "已到达目的地，万丽轩中餐厅到了，请慢用。",
            "健身房": "已到达目的地，健身房到了，请尽情使用。",
            "泳池": "已到达目的地，泳池到了，请注意安全。",
            "丝餐厅": "已到达目的地，丝餐厅到了，请慢用。",
            "卫生间": "已到达目的地，洗手间到了。",
            "大堂吧": "已到达目的地，大堂吧到了，请慢用。",
            "大门": "已到达目的地，大门到了，祝您出行愉快。",
            "礼宾台": "已到达目的地，礼宾台到了，请稍候。",
        }
        content = content_map.get(target_name)
        if content:
            if not self._play_template_audio(content):
                self.get_logger().info(f"未命中本地模板，使用在线TTS: {content}")
                self._tts_generate_and_play(content)
        else:
            self._tts_generate_and_play(f"已到达目的地，{target_name}到了。")

        

    def destroy_node(self):
        self._kws_running = False
        if self.mic_stream is not None:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except Exception:
                pass
        if hasattr(self, 'p'):
            self.p.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoiceBrainNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

