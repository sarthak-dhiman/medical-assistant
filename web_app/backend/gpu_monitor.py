import time
import pynvml
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GPU-Monitor")

def monitor_gpu():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        logger.info(f"🚀 Starting GPU Monitoring for {device_count} device(s)...")
        
        while True:
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                
                used_gb = info.used / (1024**3)
                total_gb = info.total / (1024**3)
                percent = (info.used / info.total) * 100
                
                log_msg = f"[{name}] Memory: {used_gb:.2f}GB / {total_gb:.2f}GB ({percent:.1f}%) | GPU Util: {util.gpu}%"
                
                if percent > 90:
                    logger.warning(f"⚠️ CRITICAL GPU MEMORY USAGE: {log_msg}")
                elif percent > 75:
                    logger.info(f"💡 High GPU Memory Usage: {log_msg}")
                else:
                    logger.info(log_msg)
                    
            time.sleep(10) # Log every 10 seconds
            
    except Exception as e:
        logger.error(f"❌ GPU Monitor Error: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

if __name__ == "__main__":
    monitor_gpu()
