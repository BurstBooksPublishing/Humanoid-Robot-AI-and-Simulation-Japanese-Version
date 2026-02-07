cpp
#include <cstdint>
#include <atomic>

extern int  sampleADC();          // mA 単位で返す
extern void setGate(bool on);
extern void logEvent(const char* tag, int value);

namespace {
constexpr int TRIP_MA      = 35000;
constexpr int DEBOUNCE_MS  = 5;
std::atomic<bool> latched{false};
} // namespace

void powerLoop()
{
    static int tcount = 0;

    if (latched.load(std::memory_order_acquire)) {
        return;                   // 手動リセットまで保護状態を維持
    }

    const int i_ma = sampleADC();

    if (i_ma > TRIP_MA) {
        if (++tcount >= DEBOUNCE_MS) {
            setGate(false);       // ハードウェア切断
            latched.store(true, std::memory_order_release);
            logEvent("OVERCURRENT", i_ma);
        }
    } else {
        tcount = 0;               // ノイズ期間が短ければリセット
    }
}