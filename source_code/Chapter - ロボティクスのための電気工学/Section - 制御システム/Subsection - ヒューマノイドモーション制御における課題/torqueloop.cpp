cpp
#include <chrono>
#include <cmath>
#include <array>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <hardware_interface/system_interface.hpp>

using namespace std::chrono_literals;

constexpr size_t kNumJoints = 12;
using JointArray = std::array<double, kNumJoints>;

class HighRateController : public rclcpp::Node
{
public:
  HighRateController()
  : Node("high_rate_controller"),
    clock_(std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME)),
    torque_pub_(create_publisher<std_msgs::msg::Float64MultiArray>("joint_torque_cmd", 1)),
    state_sub_(create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", 1,
        [this](const sensor_msgs::msg::JointState::SharedPtr msg) { latest_state_ = *msg; })),
    loop_timer_(create_wall_timer(1ms, std::bind(&HighRateController::controlLoop, this)))
  {
    declare_parameter("torque_limit", std::vector<double>(kNumJoints, 40.0));
    get_parameter("torque_limit", torque_limit_);
  }

private:
  void controlLoop()
  {
    const auto t0 = clock_->now();

    // センサー未受信ならスキップ
    if (latest_state_.position.empty() || latest_state_.velocity.empty()) return;

    // 状態ベクトル構築
    JointArray q{}, qdot{};
    std::copy_n(latest_state_.position.begin(), kNumJoints, q.begin());
    std::copy_n(latest_state_.velocity.begin(), kNumJoints, qdot.begin());

    // 100 Hzで目標加速度更新
    static auto last_plan = clock_->now();
    if ((t0 - last_plan) >= 10ms) {
      qdd_des_ = planner_.get_qdd_des(t0.seconds());
      last_plan = t0;
    }

    // 逆動力学フィードフォワード
    const auto tau_ff = inverseDynamics(M_, C_, g_, q, qdot, qdd_des_);

    // PDフィードバック（低ゲイン）
    JointArray tau_fb{};
    for (size_t i = 0; i < kNumJoints; ++i) {
      tau_fb[i] = Kp_ * (q_des_[i] - q[i]) + Kd_ * (qd_des_[i] - qdot[i]);
    }

    JointArray tau_cmd{};
    for (size_t i = 0; i < kNumJoints; ++i) {
      tau_cmd[i] = tau_ff[i] + tau_fb[i];
      // トルクリミット適用
      if (std::abs(tau_cmd[i]) > torque_limit_[i]) {
        tau_cmd[i] = std::copysign(torque_limit_[i], tau_cmd[i]);
      }
    }

    // 力センサオーバーロード時は即時ゼロトルク
    if (ft_sensor_.overload()) {
      tau_cmd.fill(0.0);
      estop_.raise();
    }

    // 指令送信
    auto msg = std_msgs::msg::Float64MultiArray();
    msg.data.assign(tau_cmd.begin(), tau_cmd.end());
    torque_pub_->publish(msg);

    // 遅延監視
    const auto latency = (clock_->now() - t0).seconds();
    if (latency > 0.001) RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "制御周期超過");
  }

  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr torque_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
  rclcpp::TimerBase::SharedPtr loop_timer_;

  sensor_msgs::msg::JointState latest_state_;

  struct {
    JointArray get_qdd_des(double) { return {}; }
  } planner_;
  struct {
    bool overload() { return false; }
  } ft_sensor_;
  struct {
    void raise() {}
  } estop_;

  JointArray q_des_{}, qd_des_{}, qdd_des_{};
  std::vector<double> torque_limit_;
  const double Kp_ = 20.0, Kd_ = 1.0;
  std::array<double, kNumJoints * kNumJoints> M_{}, C_{};
  JointArray g_{};
  JointArray inverseDynamics(const auto&, const auto&, const auto&,
                             const JointArray& q, const JointArray& qdot,
                             const JointArray& qdd)
  {
    JointArray tau{};
    // 簡易計算例（実機ではライブラリ使用）
    for (size_t i = 0; i < kNumJoints; ++i) {
      tau[i] = 1.0 * qdd[i] + 0.1 * qdot[i] + 0.05 * q[i];
    }
    return tau;
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HighRateController>());
  rclcpp::shutdown();
  return 0;
}