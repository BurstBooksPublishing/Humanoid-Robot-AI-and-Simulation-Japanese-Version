cpp
#include <cstdint>
#include <atomic>
#include <array>
#include <chrono>

#include "stm32f4xx_hal.h"          // HAL ベースの移植性
#include "can.h"
#include "encoder.h"
#include "pid.h"

namespace {
constexpr uint32_t  CTRL_FREQ_HZ   = 1000;          // 1 kHz 周期制御
constexpr uint32_t  CAN_TIMEOUT_MS = 2;            // CAN 送信タイムアウト
constexpr uint32_t  WD_REFRESH_WIN = CTRL_FREQ_HZ; // ウォッチドッグ更新区間

std::atomic<bool> control_flag{false};             // 割込み→メイン連携
PIDController     pid;                             // ゲインは別途初期化
Encoder           enc;                             // エンコーダ抽象層
CANBus            can;                             // CAN 抽象層

uint32_t loop_cnt = 0;                             // 周期カウンタ
} // namespace

// 1 kHz タイマ更新割込み
extern "C" void TIM1_UP_TIM10_IRQHandler() {
  if (__HAL_TIM_GET_FLAG(&htim1, TIM_FLAG_UPDATE) != RESET) {
    __HAL_TIM_CLEAR_FLAG(&htim1, TIM_FLAG_UPDATE);
    control_flag.store(true, std::memory_order_release);
  }
}

int main() {
  HAL_Init();
  SystemClock_Config();               // クロック 168 MHz 等
  MX_GPIO_Init();
  MX_TIM1_Init();                     // 1 kHz PWM ベースタイマ
  MX_CAN1_Init();
  MX_ADC1_Init();

  pid.init(0.5f, 0.01f, 0.0f);        // Kp, Ki, Kd (例)
  enc.init();
  can.init(1'000'000);                // 1 Mbps
  IWatchdog::init(10);                // 10 ms タイムアウト

  HAL_TIM_Base_Start_IT(&htim1);      // 周期割込み開始

  while (true) {
    if (control_flag.exchange(false, std::memory_order_acquire)) {
      const int32_t pos = enc.read(); // 高速、低ジッタ
      const int32_t vel = enc.differentiate(pos);
      const float   tau = pid.update(static_cast<float>(pos),
                                     static_cast<float>(vel));

      can.send_torque(tau, CAN_TIMEOUT_MS);

      if (++loop_cnt % WD_REFRESH_WIN == 0) {
        IWatchdog::refresh();         // 安全：定期リフレッシュ
      }
    }

    // 空き時間に非クリティカル処理
    can.poll_noncritical();
    Logger::maybe_flush();
  }
}