#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"
#include <math.h>

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }
constexpr double steerLimit() { return 25 * pi() / 180; }
const double eps = 1e-3;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

void transformCoordSys(const std::vector<double>& xvals, const std::vector<double>& yvals,
                       std::vector<double>& xres, std::vector<double>& yres,
                       double px, double py, double psi) {
  for (int i = 0; i < xvals.size(); ++i) {
    double x = xvals[i];
    double y = yvals[i];
    xres[i] = (x - px)*cos(-psi) - (y - py)*sin(-psi);
    yres[i] = (x - px)*sin(-psi) + (y - py)*cos(-psi);
  }
}

std::vector<double> cteNewton(Eigen::VectorXd coeffs, unsigned int iters) {
  double x = coeffs[0];
  double n, d, f, fp, fpp;

  Eigen::VectorXd f_p(3);
  f_p << coeffs[1], 2*coeffs[2], 3*coeffs[3];
  Eigen::VectorXd f_pp(2);
  f_pp << 2*coeffs[2], 6*coeffs[3];

  for (unsigned int i = 0; i < iters; ++i) {
    f = polyeval(coeffs, x);
    fp = polyeval(f_p, x);
    fpp = polyeval(f_pp, x);
    n = f*fp + x;
    d = fp*fp + fpp*f + 1;
    x -= n/d;
  }
  f = polyeval(coeffs, x);
  double distance = sqrt(f*f + x*x);
  double psides = polyeval(f_p, x);
  return {(coeffs[0] > 0) ? distance : -distance, -atan(psides)};
}


int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          double st = j[1]["steering_angle"];
          double a = j[1]["throttle"];

          /*
          * TODO: Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */

          // Transform the reference points into vehicle's coordinate system.
          std::vector<double> xTrans, yTrans;
          xTrans.resize(ptsx.size(), 0);
          yTrans.resize(ptsy.size(), 0);
          transformCoordSys(ptsx, ptsy, xTrans, yTrans, px, py, psi);

          std::cout << ptsx.size() << ", " << ptsy.size() << std::endl;
          // Solve for the fitting polynomial to the reference points.
          Eigen::VectorXd xvals = Eigen::Map<Eigen::VectorXd>(xTrans.data(), xTrans.size());
          Eigen::VectorXd yvals = Eigen::Map<Eigen::VectorXd>(yTrans.data(), yTrans.size());
          auto coeffs = polyfit(xvals, yvals, 3);

          // Construct the state vector.
          // The CTE error is calculated using Newton's method with 5 iterations.
          std::vector<double> errors = cteNewton(coeffs, 5);
          double cte = errors[0];
          double epsi = errors[1];
          Eigen::VectorXd state(6);
          // Predict the future state after 100ms due to latency.
          state[0] = v * cos(0) * 0.1;
          state[1] = v * sin(0) * 0.1;
          state[2] = (-v / 2.67) * st * 0.1;
          state[3] = v + a * 0.1;
          state[4] = cte + v*sin(epsi)*0.1;
          state[5] = epsi - (v / 2.67) * st * 0.1;


          // Solve for the optimal actuations.
          auto result = mpc.Solve(state, coeffs);

          double steer_value = result[0];
          double throttle_value = result[1];

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = -steer_value/steerLimit();
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line
          for (int i = 2; i < result.size(); i += 2) {
            mpc_x_vals.push_back(result[i]);
            mpc_y_vals.push_back(result[i+1]);
          }

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          msgJson["next_x"] = xTrans; // next_x_vals;
          msgJson["next_y"] = yTrans; // next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}