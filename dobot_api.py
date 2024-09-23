import json
import logging
import os
import socket
import threading
from copy import copy
from enum import IntEnum
from time import sleep
import re

import numpy as np

alarmControllerFile = "files/alarm_controller.json"
alarmServoFile = "files/alarm_servo.json"

# Port Feedback
FeedbackMessageDType = np.dtype([('len', np.int64,),
                   ('digital_input_bits', np.uint64,),
                   ('digital_output_bits', np.uint64,),
                   ('robot_mode', np.uint64,),
                   ('time_stamp', np.uint64),
                   ('time_stamp_reserve_bit', np.uint64,),
                   ('test_value', np.uint64,),
                   ('test_value_keep_bit', np.float64,),
                   ('speed_scaling', np.float64,),
                   ('linear_momentum_norm', np.float64,),
                   ('v_main', np.float64,),
                   ('v_robot', np.float64, ),
                   ('i_robot', np.float64,),
                   ('i_robot_keep_bit1', np.float64,),
                   ('i_robot_keep_bit2', np.float64,),
                   ('tool_accelerometer_values', np.float64, (3, )),
                   ('elbow_position', np.float64, (3, )),
                   ('elbow_velocity', np.float64, (3, )),
                   ('q_target', np.float64, (6, )),
                   ('qd_target', np.float64, (6, )),
                   ('qdd_target', np.float64, (6, )),
                   ('i_target', np.float64, (6, )),
                   ('m_target', np.float64, (6, )),
                   ('q_actual', np.float64, (6, )),
                   ('qd_actual', np.float64, (6, )),
                   ('i_actual', np.float64, (6, )),
                   ('actual_TCP_force', np.float64, (6, )),
                   ('tool_vector_actual', np.float64, (6, )),
                   ('TCP_speed_actual', np.float64, (6, )),
                   ('TCP_force', np.float64, (6, )),
                   ('Tool_vector_target', np.float64, (6, )),
                   ('TCP_speed_target', np.float64, (6, )),
                   ('motor_temperatures', np.float64, (6, )),
                   ('joint_modes', np.float64, (6, )),
                   ('v_actual', np.float64, (6, )),
                   ('hand_type', np.byte, (4, )),
                   ('user', np.byte,),
                   ('tool', np.byte,),
                   ('run_queued_cmd', np.byte,),
                   ('pause_cmd_flag', np.byte,),
                   ('velocity_ratio', np.byte,),
                   ('acceleration_ratio', np.byte,),
                   ('jerk_ratio', np.byte,),
                   ('xyz_velocity_ratio', np.byte,),
                   ('r_velocity_ratio', np.byte,),
                   ('xyz_acceleration_ratio', np.byte,),
                   ('r_acceleration_ratio', np.byte,),
                   ('xyz_jerk_ratio', np.byte,),
                   ('r_jerk_ratio', np.byte,),
                   ('brake_status', np.byte,),
                   ('enable_status', np.byte,),
                   ('drag_status', np.byte,),
                   ('running_status', np.byte,),
                   ('error_status', np.byte,),
                   ('jog_status', np.byte,),
                   ('robot_type', np.byte,),
                   ('drag_button_signal', np.byte,),
                   ('enable_button_signal', np.byte,),
                   ('record_button_signal', np.byte,),
                   ('reappear_button_signal', np.byte,),
                   ('jaw_button_signal', np.byte,),
                   ('six_force_online', np.byte,),
                   ('reserve2', np.byte, (66, )),
                   ('vibrationdisZ', np.float64,),
                   ('currentcommandid', np.uint64,),
                   ('m_actual', np.float64, (6, )),
                   ('load', np.float64,),
                   ('center_x', np.float64,),
                   ('center_y', np.float64,),
                   ('center_z', np.float64,),
                   ('user[6]', np.float64, (6, )),
                   ('tool[6]', np.float64, (6, )),
                   ('trace_index', np.float64,),
                   ('six_force_value', np.float64, (6, )),
                   ('target_quaternion', np.float64, (4, )),
                   ('actual_quaternion', np.float64, (4, )),
                   ('auto_manual_mode', np.byte, (2,)),
                   ('reserve3', np.byte, (22, ))])

IPStr = str
PortInt = int

TCP_PORTS = (29999, 30003, 30004, 30005)  # TCP communication interface ports

# 读取控制器和伺服告警文件


def alarmAlarmJsonFile():
    currrntDirectory = os.path.dirname(__file__)
    jsonContrellorPath = os.path.join(currrntDirectory, alarmControllerFile)
    jsonServoPath = os.path.join(currrntDirectory, alarmServoFile)

    with open(jsonContrellorPath, encoding='utf-8') as f:
        dataController = json.load(f)
    with open(jsonServoPath, encoding='utf-8') as f:
        dataServo = json.load(f)
    return dataController, dataServo


class RobotModes(IntEnum):
    """Robot modes."""
    INIT = 1
    BRAKE_OPEN = 2
    POWER_STATUS = 3
    DISABLED = 4
    ENABLE = 5
    BACKDRIVE = 6
    RUNNING = 7
    RECORDING = 8
    ERROR = 9
    PAUSE = 10
    JOG = 11


class DobotApiBase:
    """Base class for Dobot API"""

    def __init__(self, ip: IPStr, port: PortInt, logger: logging.Logger):
        """
        Initialize the Dobot API.
        :param ip: ip address
        :param port: port number
        :param logger: class logger
        """
        if port not in TCP_PORTS:
            raise ValueError(f"Invalid port: {port}. Must be one of {TCP_PORTS}")

        self.ip = ip
        self.port = port
        self.socket_dobot = 0
        self.__global_lock = threading.Lock()
        self._logger = logger

        try:
            self.socket_dobot = socket.socket()
            self.socket_dobot.connect((self.ip, self.port))
        except socket.error:
            raise RuntimeError(f"Could not connect to socket: {socket.error}")

    def send_data(self, string: str):
        """Send data to the socket. Retries until successful."""
        self._logger.debug(f"Send to {self.ip}:{self.port}: {string}")
        try:
            self.socket_dobot.send(str.encode(string, "utf-8"))
        except Exception as e:
            self._logger.warning(f"Exception raised when sending data: {e}. Retrying (potentially forever)...")
            while True:
                try:
                    self.socket_dobot = self.reconnect(self.ip, self.port)
                    self.socket_dobot.send(str.encode(string, "utf-8"))
                    break
                except Exception:
                    sleep(1)

    def wait_reply(self):
        """
        Read the return value
        """
        data = ""
        try:
            data = self.socket_dobot.recv(1024)
        except Exception as e:
            self._logger.warning(e)
            self.socket_dobot = self.reconnect(self.ip, self.port)

        finally:
            data_str = str(data, encoding="utf-8")
            self._logger.debug(f"Receive from {self.ip}:{self.port}: {data_str}")
            return data_str

    def close(self) -> None:
        """
        Close the port
        """
        if self.socket_dobot != 0:
            try:
                self.socket_dobot.shutdown(socket.SHUT_RDWR)
                self.socket_dobot.close()
            except socket.error as e:
                self._logger.warning(f"Error while closing socket: {e}")

    def send_receive_msg(self, string: str) -> str:
        """
        Send data to the robot and wait for the reply. Returns the reply.
        """
        with self.__global_lock:
            self.send_data(string)
            recv_data = self.wait_reply()
            self._logger.info(f'recv_data type: {type(recv_data)}')
            self.parse_result_id(recv_data)
            return recv_data

    def __del__(self) -> None:
        """Close the port when the object is deleted"""
        self.close()

    def reconnect(self, ip: IPStr, port: PortInt):
        """Reconnect to the robot"""
        while True:
            try:
                socket_dobot = socket.socket()
                socket_dobot.connect((ip, port))
                break
            except Exception:
                sleep(1)
        return socket_dobot

    def parse_result_id(self, value_recv: str) -> None:
        """
        Parse the returned string and print in human-readable format
        :param recv_data: value returned by the robot
        """
        if value_recv.find("Not Tcp") != -1:  # Judge whether the robot is in TCP mode by the return value
            self._logger.warning("Control mode is not TCP")
            return

        print(f'value received: {value_recv}')
        recv_data = re.findall(r"-?\d+", value_recv)
        recv_data = [int(num) for num in recv_data]

        if not recv_data:
            self._logger.warning("No values received")
            return

        error_id = recv_data[0]

        if error_id == 0:
            return  # No error, normal operation
        elif error_id == -1:
            self._logger.warning("Command execution failed")
        elif error_id == -2:
            self._logger.warning("The robot is in an error state")
        elif error_id == -3:
            self._logger.warning("The robot is in an emergency stop state")
        elif error_id == -4:
            self._logger.warning("The robot is in a power down state")
        else:
            self._logger.warning(f"Unexpected Error ID: {error_id}")


class DobotApiDashboard(DobotApiBase):
    """
    Define class dobot_api_dashboard to establish a connection to Dobot
    """
    def __init__(self, ip: IPStr, port: PortInt, logger: logging.Logger):
        super().__init__(ip=ip, port=port, logger=logger)

    def EnableRobot(self):
        """
        Enable the robot
        """
        string = "EnableRobot()"
        return self.send_receive_msg(string)

    def DisableRobot(self):
        """
        Disabled the robot
        """
        string = "DisableRobot()"
        return self.send_receive_msg(string)

    def ClearError(self):
        """
        Clear controller alarm information
        """
        string = "ClearError()"
        return self.send_receive_msg(string)

    def ResetRobot(self):
        """
        Robot stop
        """
        string = "ResetRobot()"
        return self.send_receive_msg(string)

    def SpeedFactor(self, speed):
        """
        Setting the Global rate   
        speed:Rate value(Value range:1~100)
        """
        string = "SpeedFactor({:d})".format(speed)
        return self.send_receive_msg(string)

    def User(self, index):
        """
        Select the calibrated user coordinate system
        index : Calibrated index of user coordinates
        """
        string = "User({:d})".format(index)
        return self.send_receive_msg(string)

    def Tool(self, index):
        """
        Select the calibrated tool coordinate system
        index : Calibrated index of tool coordinates
        """
        string = "Tool({:d})".format(index)
        return self.send_receive_msg(string)

    def RobotMode(self):
        """
        View the robot status
        """
        string = "RobotMode()"
        return self.send_receive_msg(string)

    def PayLoad(self, weight, inertia):
        """
        Setting robot load
        weight : The load weight
        inertia: The load moment of inertia
        """
        string = "PayLoad({:f},{:f})".format(weight, inertia)
        return self.send_receive_msg(string)

    def DO(self, index, status):
        """
        Set digital signal output (Queue instruction)
        index : Digital output index (Value range:1~24)
        status : Status of digital signal output port(0:Low level,1:High level
        """
        string = "DO({:d},{:d})".format(index, status)
        return self.send_receive_msg(string)

    def DOExecute(self, index, status):
        """
        Set digital signal output (Instructions immediately)
        index : Digital output index (Value range:1~24)
        status : Status of digital signal output port(0:Low level,1:High level)
        """
        string = "DOExecute({:d},{:d})".format(index, status)
        return self.send_receive_msg(string)

    def ToolDO(self, index, status):
        """
        Set terminal signal output (Queue instruction)
        index : Terminal output index (Value range:1~2)
        status : Status of digital signal output port(0:Low level,1:High level)
        """
        string = "ToolDO({:d},{:d})".format(index, status)
        return self.send_receive_msg(string)

    def ToolDOExecute(self, index, status):
        """
        Set terminal signal output (Instructions immediately)
        index : Terminal output index (Value range:1~2)
        status : Status of digital signal output port(0:Low level,1:High level)
        """
        string = "ToolDOExecute({:d},{:d})".format(index, status)
        return self.send_receive_msg(string)

    def AO(self, index, val):
        """
        Set analog signal output (Queue instruction)
        index : Analog output index (Value range:1~2)
        val : Voltage value (0~10)
        """
        string = "AO({:d},{:f})".format(index, val)
        return self.send_receive_msg(string)

    def AOExecute(self, index, val):
        """
        Set analog signal output (Instructions immediately)
        index : Analog output index (Value range:1~2)
        val : Voltage value (0~10)
        """
        string = "AOExecute({:d},{:f})".format(index, val)
        return self.send_receive_msg(string)

    def AccJ(self, speed):
        """
        Set joint acceleration ratio (Only for MovJ, MovJIO, MovJR, JointMovJ commands)
        speed : Joint acceleration ratio (Value range:1~100)
        """
        string = "AccJ({:d})".format(speed)
        return self.send_receive_msg(string)

    def AccL(self, speed):
        """
        Set the coordinate system acceleration ratio (Only for MovL, MovLIO, MovLR, Jump, Arc, Circle commands)
        speed : Cartesian acceleration ratio (Value range:1~100)
        """
        string = "AccL({:d})".format(speed)
        return self.send_receive_msg(string)

    def SpeedJ(self, speed):
        """
        Set joint speed ratio (Only for MovJ, MovJIO, MovJR, JointMovJ commands)
        speed : Joint velocity ratio (Value range:1~100)
        """
        string = "SpeedJ({:d})".format(speed)
        return self.send_receive_msg(string)

    def SpeedL(self, speed):
        """
        Set the cartesian acceleration ratio (Only for MovL, MovLIO, MovLR, Jump, Arc, Circle commands)
        speed : Cartesian acceleration ratio (Value range:1~100)
        """
        string = "SpeedL({:d})".format(speed)
        return self.send_receive_msg(string)

    def Arch(self, index):
        """
        Set the Jump gate parameter index (This index contains: start point lift height, maximum lift height, end point drop height)
        index : Parameter index (Value range:0~9)
        """
        string = "Arch({:d})".format(index)
        return self.send_receive_msg(string)

    def CP(self, ratio):
        """
        Set smooth transition ratio
        ratio : Smooth transition ratio (Value range:1~100)
        """
        string = "CP({:d})".format(ratio)
        return self.send_receive_msg(string)

    def LimZ(self, value):
        """
        Set the maximum lifting height of door type parameters
        value : Maximum lifting height (Highly restricted:Do not exceed the limit position of the z-axis of the manipulator)
        """
        string = "LimZ({:d})".format(value)
        return self.send_receive_msg(string)

    def SetArmOrientation(self, r, d, n, cfg):
        """
        Set the hand command
        r : Mechanical arm direction, forward/backward (1:forward -1:backward)
        d : Mechanical arm direction, up elbow/down elbow (1:up elbow -1:down elbow)
        n : Whether the wrist of the mechanical arm is flipped (1:The wrist does not flip -1:The wrist flip)
        cfg :Sixth axis Angle identification
            (1, - 2... : Axis 6 Angle is [0,-90] is -1; [90, 180] - 2; And so on
            1, 2... : axis 6 Angle is [0,90] is 1; [90180] 2; And so on)
        """
        string = "SetArmOrientation({:d},{:d},{:d},{:d})".format(r, d, n, cfg)
        return self.send_receive_msg(string)

    def PowerOn(self):
        """
        Powering on the robot
        Note: It takes about 10 seconds for the robot to be enabled after it is powered on.
        """
        string = "PowerOn()"
        return self.send_receive_msg(string)

    def RunScript(self, project_name):
        """
        Run the script file
        project_name :Script file name
        """
        string = "RunScript({:s})".format(project_name)
        return self.send_receive_msg(string)

    def StopScript(self):
        """
        Stop scripts
        """
        string = "StopScript()"
        return self.send_receive_msg(string)

    def PauseScript(self):
        """
        Pause the script
        """
        string = "PauseScript()"
        return self.send_receive_msg(string)

    def ContinueScript(self):
        """
        Continue running the script
        """
        string = "ContinueScript()"
        return self.send_receive_msg(string)

    def GetHoldRegs(self, id, addr, count, type):
        """
        Read hold register
        id :Secondary device NUMBER (A maximum of five devices can be supported. The value ranges from 0 to 4
            Set to 0 when accessing the internal slave of the controller)
        addr :Hold the starting address of the register (Value range:3095~4095)
        count :Reads the specified number of types of data (Value range:1~16)
        type :The data type
            If null, the 16-bit unsigned integer (2 bytes, occupying 1 register) is read by default
            "U16" : reads 16-bit unsigned integers (2 bytes, occupying 1 register)
            "U32" : reads 32-bit unsigned integers (4 bytes, occupying 2 registers)
            "F32" : reads 32-bit single-precision floating-point number (4 bytes, occupying 2 registers)
            "F64" : reads 64-bit double precision floating point number (8 bytes, occupying 4 registers)
        """
        string = "GetHoldRegs({:d},{:d},{:d},{:s})".format(
            id, addr, count, type)
        return self.send_receive_msg(string)

    def SetHoldRegs(self, id, addr, count, table, type=None):
        """
        Write hold register
        id :Secondary device NUMBER (A maximum of five devices can be supported. The value ranges from 0 to 4
            Set to 0 when accessing the internal slave of the controller)
        addr :Hold the starting address of the register (Value range:3095~4095)
        count :Writes the specified number of types of data (Value range:1~16)
        type :The data type
            If null, the 16-bit unsigned integer (2 bytes, occupying 1 register) is read by default
            "U16" : reads 16-bit unsigned integers (2 bytes, occupying 1 register)
            "U32" : reads 32-bit unsigned integers (4 bytes, occupying 2 registers)
            "F32" : reads 32-bit single-precision floating-point number (4 bytes, occupying 2 registers)
            "F64" : reads 64-bit double precision floating point number (8 bytes, occupying 4 registers)
        """
        if type is not None:
            string = "SetHoldRegs({:d},{:d},{:d},{:s},{:s})".format(
                id, addr, count, table, type)
        else:
            string = "SetHoldRegs({:d},{:d},{:d},{:s})".format(
                id, addr, count, table)
        return self.send_receive_msg(string)

    def GetErrorID(self):
        """
        Get robot error code
        """
        string = "GetErrorID()"
        return self.send_receive_msg(string)

    def DOExecute(self, offset1, offset2):
        string = "DOExecute({:d},{:d}".format(offset1, offset2)+")"
        return self.send_receive_msg(string)

    def ToolDO(self, offset1, offset2):
        string = "ToolDO({:d},{:d}".format(offset1, offset2)+")"
        return self.send_receive_msg(string)

    def ToolDOExecute(self, offset1, offset2):
        string = "ToolDOExecute({:d},{:d}".format(offset1, offset2)+")"
        return self.send_receive_msg(string)

    def SetArmOrientation(self, offset1):
        string = "SetArmOrientation({:d}".format(offset1)+")"
        return self.send_receive_msg(string)

    def SetPayload(self, offset1, *dynParams):
        string = "SetPayload({:f}".format(
            offset1)
        for params in dynParams:
            string = string + str(params)+","
        string = string + ")"
        return self.send_receive_msg(string)

    def PositiveSolution(self, offset1, offset2, offset3, offset4, offset5, offset6, user, tool):
        string = "PositiveSolution({:f},{:f},{:f},{:f},{:f},{:f},{:d},{:d}".format(
            offset1, offset2, offset3, offset4, offset5, offset6, user, tool)+")"
        return self.send_receive_msg(string)

    def InverseSolution(self, offset1, offset2, offset3, offset4, offset5, offset6, user, tool, *dynParams):
        string = "InverseSolution({:f},{:f},{:f},{:f},{:f},{:f},{:d},{:d}".format(
            offset1, offset2, offset3, offset4, offset5, offset6, user, tool)
        for params in dynParams:
            self._logger.info(type(params), params)
            string = string + repr(params)
        string = string + ")"
        return self.send_receive_msg(string)

    def SetCollisionLevel(self, offset1):
        string = "SetCollisionLevel({:d}".format(offset1)+")"
        return self.send_receive_msg(string)

    def GetAngle(self):
        string = "GetAngle()"
        return self.send_receive_msg(string)

    def GetPose(self):
        string = "GetPose()"
        return self.send_receive_msg(string)

    def EmergencyStop(self):
        string = "EmergencyStop()"
        return self.send_receive_msg(string)

    def ModbusCreate(self, ip, port, slave_id, isRTU):
        string = "ModbusCreate({:s},{:d},{:d},{:d}".format(
            ip, port, slave_id, isRTU)+")"
        return self.send_receive_msg(string)

    def ModbusClose(self, offset1):
        string = "ModbusClose({:d}".format(offset1)+")"
        return self.send_receive_msg(string)

    def SetSafeSkin(self, offset1):
        string = "SetSafeSkin({:d}".format(offset1)+")"
        return self.send_receive_msg(string)

    def SetObstacleAvoid(self, offset1):
        string = "SetObstacleAvoid({:d}".format(offset1)+")"
        return self.send_receive_msg(string)

    def GetTraceStartPose(self, offset1):
        string = "GetTraceStartPose({:s}".format(offset1)+")"
        return self.send_receive_msg(string)

    def GetPathStartPose(self, offset1):
        string = "GetPathStartPose({:s}".format(offset1)+")"
        return self.send_receive_msg(string)

    def HandleTrajPoints(self, offset1):
        string = "HandleTrajPoints({:s}".format(offset1)+")"
        return self.send_receive_msg(string)

    def GetSixForceData(self):
        string = "GetSixForceData()"
        return self.send_receive_msg(string)

    def SetCollideDrag(self, offset1):
        string = "SetCollideDrag({:d}".format(offset1)+")"
        return self.send_receive_msg(string)

    def SetTerminalKeys(self, offset1):
        string = "SetTerminalKeys({:d}".format(offset1)+")"
        return self.send_receive_msg(string)

    def SetTerminal485(self, offset1, offset2, offset3, offset4):
        string = "SetTerminal485({:d},{:d},{:s},{:d}".format(
            offset1, offset2, offset3, offset4)+")"
        return self.send_receive_msg(string)

    def GetTerminal485(self):
        string = "GetTerminal485()"
        return self.send_receive_msg(string)

    def TCPSpeed(self, offset1):
        string = "TCPSpeed({:d}".format(offset1)+")"
        return self.send_receive_msg(string)

    def TCPSpeedEnd(self):
        string = "TCPSpeedEnd()"
        return self.send_receive_msg(string)

    def GetInBits(self, offset1, offset2, offset3):
        string = "GetInBits({:d},{:d},{:d}".format(
            offset1, offset2, offset3)+")"
        return self.send_receive_msg(string)

    def GetInRegs(self, offset1, offset2, offset3, *dynParams):
        string = "GetInRegs({:d},{:d},{:d}".format(offset1, offset2, offset3)
        for params in dynParams:
            self._logger.info(type(params), params)
            string = string + params[0]
        string = string + ")"
        return self.send_receive_msg(string)

    def GetCoils(self, offset1, offset2, offset3):
        string = "GetCoils({:d},{:d},{:d}".format(
            offset1, offset2, offset3)+")"
        return self.send_receive_msg(string)

    def SetCoils(self, offset1, offset2, offset3, offset4):
        string = "SetCoils({:d},{:d},{:d}".format(
            offset1, offset2, offset3)+"," + repr(offset4)+")"
        self._logger.info(str(offset4))
        return self.send_receive_msg(string)

    def DI(self, offset1):
        string = "DI({:d}".format(offset1)+")"
        return self.send_receive_msg(string)

    def ToolDI(self, offset1):
        string = "DI({:d}".format(offset1)+")"
        return self.send_receive_msg(string)

    def DOGroup(self, *dynParams):
        string = "DOGroup("
        for params in dynParams:
            string = string + str(params)+","
        string = string + ")"
        self._logger.info(string)
        return self.wait_reply()

    def BrakeControl(self, offset1, offset2):
        string = "BrakeControl({:d},{:d}".format(offset1, offset2)+")"
        return self.send_receive_msg(string)

    def StartDrag(self):
        string = "StartDrag()"
        return self.send_receive_msg(string)

    def StopDrag(self):
        string = "StopDrag()"
        return self.send_receive_msg(string)

    def LoadSwitch(self, offset1):
        string = "LoadSwitch({:d}".format(offset1)+")"
        return self.send_receive_msg(string)

    def wait(self,t):
        string = "wait({:d})".format(t)
        return self.send_receive_msg(string)

    def pause(self):
        string = "pause()"
        return self.send_receive_msg(string)

    def Continue(self):
        string = "continue()"
        return self.send_receive_msg(string)


class DobotApiMove(DobotApiBase):
    """
    Define class dobot_api_move to establish a connection to Dobot
    """
    def __init__(self, ip: IPStr, port: PortInt, logger: logging.Logger):
        super().__init__(ip=ip, port=port, logger=logger)

    def MovJ(self, x, y, z, rx, ry, rz, *dynParams):
        """
        Joint motion interface (point-to-point motion mode)
        x: A number in the Cartesian coordinate system x
        y: A number in the Cartesian coordinate system y
        z: A number in the Cartesian coordinate system z
        rx: Position of Rx axis in Cartesian coordinate system
        ry: Position of Ry axis in Cartesian coordinate system
        rz: Position of Rz axis in Cartesian coordinate system
        """
        string = "MovJ({:f},{:f},{:f},{:f},{:f},{:f}".format(
            x, y, z, rx, ry, rz)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        self._logger.info(string)
        return self.send_receive_msg(string)

    def MovL(self, x, y, z, rx, ry, rz, *dynParams):
        """
        Coordinate system motion interface (linear motion mode)
        x: A number in the Cartesian coordinate system x
        y: A number in the Cartesian coordinate system y
        z: A number in the Cartesian coordinate system z
        rx: Position of Rx axis in Cartesian coordinate system
        ry: Position of Ry axis in Cartesian coordinate system
        rz: Position of Rz axis in Cartesian coordinate system
        """
        string = "MovL({:f},{:f},{:f},{:f},{:f},{:f}".format(
            x, y, z, rx, ry, rz)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        self._logger.info(string)
        return self.send_receive_msg(string)

    def JointMovJ(self, j1, j2, j3, j4, j5, j6, *dynParams):
        """
        Joint motion interface (linear motion mode)
        j1~j6:Point position values on each joint
        """
        string = "JointMovJ({:f},{:f},{:f},{:f},{:f},{:f}".format(
            j1, j2, j3, j4, j5, j6)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.send_receive_msg(string)

    def Jump(self):
        self._logger.info("待定")

    def RelMovJ(self, offset1, offset2, offset3, offset4, offset5, offset6, *dynParams):
        """
        Offset motion interface (point-to-point motion mode)
        j1~j6:Point position values on each joint
        """
        string = "RelMovJ({:f},{:f},{:f},{:f},{:f},{:f}".format(
            offset1, offset2, offset3, offset4, offset5, offset6)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.send_receive_msg(string)

    def RelMovL(self, offsetX, offsetY, offsetZ, *dynParams):
        """
        Offset motion interface (point-to-point motion mode)
        x: Offset in the Cartesian coordinate system x
        y: offset in the Cartesian coordinate system y
        z: Offset in the Cartesian coordinate system Z
        """
        string = "RelMovL({:f},{:f},{:f}".format(offsetX, offsetY, offsetZ)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.send_receive_msg(string)

    def MovLIO(self, x, y, z, a, b, c, *dynParams):
        """
        Set the digital output port state in parallel while moving in a straight line
        x: A number in the Cartesian coordinate system x
        y: A number in the Cartesian coordinate system y
        z: A number in the Cartesian coordinate system z
        a: A number in the Cartesian coordinate system a
        b: A number in the Cartesian coordinate system b
        c: a number in the Cartesian coordinate system c
        *dynParams :Parameter Settings（Mode、Distance、Index、Status）
                    Mode :Set Distance mode (0: Distance percentage; 1: distance from starting point or target point)
                    Distance :Runs the specified distance（If Mode is 0, the value ranges from 0 to 100；When Mode is 1, if the value is positive,
                             it indicates the distance from the starting point. If the value of Distance is negative, it represents the Distance from the target point）
                    Index :Digital output index （Value range:1~24）
                    Status :Digital output state（Value range:0/1）
        """
        # example: MovLIO(0,50,0,0,0,0,(0,50,1,0),(1,1,2,1))
        string = "MovLIO({:f},{:f},{:f},{:f},{:f},{:f}".format(
            x, y, z, a, b, c)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.send_receive_msg(string)

    def MovJIO(self, x, y, z, a, b, c, *dynParams):
        """
        Set the digital output port state in parallel during point-to-point motion
        x: A number in the Cartesian coordinate system x
        y: A number in the Cartesian coordinate system y
        z: A number in the Cartesian coordinate system z
        a: A number in the Cartesian coordinate system a
        b: A number in the Cartesian coordinate system b
        c: a number in the Cartesian coordinate system c
        *dynParams :Parameter Settings（Mode、Distance、Index、Status）
                    Mode :Set Distance mode (0: Distance percentage; 1: distance from starting point or target point)
                    Distance :Runs the specified distance（If Mode is 0, the value ranges from 0 to 100；When Mode is 1, if the value is positive,
                             it indicates the distance from the starting point. If the value of Distance is negative, it represents the Distance from the target point）
                    Index :Digital output index （Value range:1~24）
                    Status :Digital output state（Value range:0/1）
        """
        # example: MovJIO(0,50,0,0,0,0,(0,50,1,0),(1,1,2,1))
        string = "MovJIO({:f},{:f},{:f},{:f},{:f},{:f}".format(
            x, y, z, a, b, c)
        self.log("Send to 192.168.5.1:29999:" + string)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.send_receive_msg(string)

    def Arc(self, x1, y1, z1, a1, b1, c1, x2, y2, z2, a2, b2, c2, *dynParams):
        """
        Circular motion instruction
        x1, y1, z1, a1, b1, c1 :Is the point value of intermediate point coordinates
        x2, y2, z2, a2, b2, c2 :Is the value of the end point coordinates
        Note: This instruction should be used together with other movement instructions
        """
        string = "Arc({:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(
            x1, y1, z1, a1, b1, c1, x2, y2, z2, a2, b2, c2)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.send_receive_msg(string)

    def Circle3(self, x1, y1, z1, a1, b1, c1, x2, y2, z2, a2, b2, c2, count,*dynParams):
        """
        Full circle motion command
        count:Run laps
        x1, y1, z1, r1 :Is the point value of intermediate point coordinates
        x2, y2, z2, r2 :Is the value of the end point coordinates
        Note: This instruction should be used together with other movement instructions
        """
        string = "Circle3({:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:d}".format(
            x1, y1, z1, a1, b1, c1, x2, y2, z2, a2, b2, c2, count)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.send_receive_msg(string)

    def ServoJ(self, j1, j2, j3, j4, j5, j6,t=0.1,lookahead_time=50,gain=500):
        """
        Dynamic follow command based on joint space
        j1~j6:Point position values on each joint
        
        可选参数:t、lookahead_time、gain
        t float 该点位的运行时间,默认0.1,单位:s  取值范围:[0.02,3600.0]
        lookahead_time   float 作用类似于PID的D项,默认50,标量,无单位 取值范围:[20.0,100.0]
        gain float   目标位置的比例放大器,作用类似于PID的P项,  默认500,标量,无单位   取值范围:[200.0,1000.0]
        """
        string = "ServoJ({:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f})".format(
            j1, j2, j3, j4, j5, j6,t,lookahead_time,gain)
        return self.send_receive_msg(string)

    def ServoJS(self, j1, j2, j3, j4, j5, j6):
        """
        功能:基于关节空间的动态跟随运动。
        格式:ServoJS(J1,J2,J3,J4,J5,J6)
        """
        string = "ServoJS({:f},{:f},{:f},{:f},{:f},{:f})".format(
            j1, j2, j3, j4, j5, j6)
        return self.send_receive_msg(string)

    def ServoP(self, x, y, z, a, b, c):
        """
        Dynamic following command based on Cartesian space
        x, y, z, a, b, c :Cartesian coordinate point value
        """
        string = "ServoP({:f},{:f},{:f},{:f},{:f},{:f})".format(
            x, y, z, a, b, c)
        return self.send_receive_msg(string)

    def MoveJog(self, axis_id, *dynParams):
        """
        Joint motion
        axis_id: Joint motion axis, optional string value:
            J1+ J2+ J3+ J4+ J5+ J6+
            J1- J2- J3- J4- J5- J6- 
            X+ Y+ Z+ Rx+ Ry+ Rz+ 
            X- Y- Z- Rx- Ry- Rz-
        *dynParams: Parameter Settings（coord_type, user_index, tool_index）
                    coord_type: 1: User coordinate 2: tool coordinate (default value is 1)
                    user_index: user index is 0 ~ 9 (default value is 0)
                    tool_index: tool index is 0 ~ 9 (default value is 0)
        """
        string = "MoveJog({:s}".format(axis_id)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.send_receive_msg(string)

    def StartTrace(self, trace_name):
        """
        Trajectory fitting (track file Cartesian points)
        trace_name: track file name (including suffix)
        (The track path is stored in /dobot/userdata/project/process/trajectory/)

        It needs to be used together with `GetTraceStartPose(recv_string.json)` interface
        """
        string = f"StartTrace({trace_name})"
        return self.send_receive_msg(string)

    def StartPath(self, trace_name, const, cart):
        """
        Track reproduction. (track file joint points)
        trace_name: track file name (including suffix)
        (The track path is stored in /dobot/userdata/project/process/trajectory/)
        const: When const = 1, it repeats at a constant speed, and the pause and dead zone in the track will be removed;
               When const = 0, reproduce according to the original speed;
        cart: When cart = 1, reproduce according to Cartesian path;
              When cart = 0, reproduce according to the joint path;

        It needs to be used together with `GetTraceStartPose(recv_string.json)` interface
        """
        string = f"StartPath({trace_name}, {const}, {cart})"
        return self.send_receive_msg(string)

    def StartFCTrace(self, trace_name):
        """
        Trajectory fitting with force control. (track file Cartesian points)
        trace_name: track file name (including suffix)
        (The track path is stored in /dobot/userdata/project/process/trajectory/)

        It needs to be used together with `GetTraceStartPose(recv_string.json)` interface
        """
        string = f"StartFCTrace({trace_name})"
        return self.send_receive_msg(string)

    def Sync(self):
        """
        The blocking program executes the queue instruction and returns after all the queue instructions are executed
        """
        string = "Sync()"
        return self.send_receive_msg(string)

    def RelMovJTool(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, tool, *dynParams):
        """
        The relative motion command is carried out along the tool coordinate system, and the end motion mode is joint motion
        offset_x: X-axis direction offset
        offset_y: Y-axis direction offset
        offset_z: Z-axis direction offset
        offset_rx: Rx axis position
        offset_ry: Ry axis position
        offset_rz: Rz axis position
        tool: Select the calibrated tool coordinate system, value range: 0 ~ 9
        *dynParams: parameter Settings（speed_j, acc_j, user）
                    speed_j: Set joint speed scale, value range: 1 ~ 100
                    acc_j: Set acceleration scale value, value range: 1 ~ 100
                    user: Set user coordinate system index
        """
        string = "RelMovJTool({:f},{:f},{:f},{:f},{:f},{:f}, {:d}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, tool)
        for params in dynParams:
            self._logger.info(type(params), params)
            string = string + ", SpeedJ={:d}, AccJ={:d}, User={:d}".format(
                params[0], params[1], params[2])
        string = string + ")"
        return self.send_receive_msg(string)

    def RelMovLTool(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, tool, *dynParams):
        """
        Carry out relative motion command along the tool coordinate system, and the end motion mode is linear motion
        offset_x: X-axis direction offset
        offset_y: Y-axis direction offset
        offset_z: Z-axis direction offset
        offset_rx: Rx axis position
        offset_ry: Ry axis position
        offset_rz: Rz axis position
        tool: Select the calibrated tool coordinate system, value range: 0 ~ 9
        *dynParams: parameter Settings（speed_l, acc_l, user）
                    speed_l: Set Cartesian speed scale, value range: 1 ~ 100
                    acc_l: Set acceleration scale value, value range: 1 ~ 100
                    user: Set user coordinate system index
        """
        string = "RelMovLTool({:f},{:f},{:f},{:f},{:f},{:f}, {:d}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, tool)
        for params in dynParams:
            self._logger.info(type(params), params)
            string = string + ", SpeedJ={:d}, AccJ={:d}, User={:d}".format(
                params[0], params[1], params[2])
        string = string + ")"
        return self.send_receive_msg(string)

    def RelMovJUser(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user, *dynParams):
        """
        The relative motion command is carried out along the user coordinate system, and the end motion mode is joint motion
        offset_x: X-axis direction offset
        offset_y: Y-axis direction offset
        offset_z: Z-axis direction offset
        offset_rx: Rx axis position
        offset_ry: Ry axis position
        offset_rz: Rz axis position

        user: Select the calibrated user coordinate system, value range: 0 ~ 9
        *dynParams: parameter Settings（speed_j, acc_j, tool）
                    speed_j: Set joint speed scale, value range: 1 ~ 100
                    acc_j: Set acceleration scale value, value range: 1 ~ 100
                    tool: Set tool coordinate system index
        """
        string = "RelMovJUser({:f},{:f},{:f},{:f},{:f},{:f}, {:d}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.send_receive_msg(string)

    def RelMovLUser(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user, *dynParams):
        """
        The relative motion command is carried out along the user coordinate system, and the end motion mode is linear motion
        offset_x: X-axis direction offset
        offset_y: Y-axis direction offset
        offset_z: Z-axis direction offset
        offset_rx: Rx axis position
        offset_ry: Ry axis position
        offset_rz: Rz axis position
        user: Select the calibrated user coordinate system, value range: 0 ~ 9
        *dynParams: parameter Settings（speed_l, acc_l, tool）
                    speed_l: Set Cartesian speed scale, value range: 1 ~ 100
                    acc_l: Set acceleration scale value, value range: 1 ~ 100
                    tool: Set tool coordinate system index
        """
        string = "RelMovLUser({:f},{:f},{:f},{:f},{:f},{:f}, {:d}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.send_receive_msg(string)

    def RelJointMovJ(self, offset1, offset2, offset3, offset4, offset5, offset6, *dynParams):
        """
        The relative motion command is carried out along the joint coordinate system of each axis, and the end motion mode is joint motion
        Offset motion interface (point-to-point motion mode)
        j1~j6:Point position values on each joint
        *dynParams: parameter Settings（speed_j, acc_j, user）
                    speed_j: Set Cartesian speed scale, value range: 1 ~ 100
                    acc_j: Set acceleration scale value, value range: 1 ~ 100
        """
        string = "RelJointMovJ({:f},{:f},{:f},{:f},{:f},{:f}".format(
            offset1, offset2, offset3, offset4, offset5, offset6)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.send_receive_msg(string)


class RobotStatusItem:
    """Class for storing robot status data."""

    def __init__(self) -> None:
        self.len = 0
        self.digital_input_bits = 0
        self.digital_output_bits = 0
        self.robot_mode = 0
        self.timestamp = 0
        self.timestamp_reserve_bit = 0
        self.test_value = 0
        self.test_value_keep_bit = 0.0
        self.speed_scaling = 0.0
        self.linear_momentum_norm = 0.0
        self.v_main = 0.0
        self.v_robot = 0.0
        self.i_robot = 0.0
        self.i_robot_keep_bit1 = 0.0
        self.i_robot_keep_bit2 = 0.0
        self.tool_accelerometer_values = []
        self.elbow_position = []
        self.elbow_velocity = []
        self.q_target = []
        self.qd_target = []
        self.qdd_target = []
        self.i_target = []
        self.m_target = []
        self.q_actual = []
        self.qd_actual = []
        self.i_actual = []
        self.actual_tcp_force = []
        self.tool_vector_actual = []
        self.tcp_speed_actual = []
        self.tcp_force = []
        self.tool_vector_target = []
        self.tcp_speed_target = []
        self.motor_temperatures = []
        self.joint_modes = []
        self.v_actual = []
        self.hand_type = []
        self.user = 0
        self.tool = 0
        self.run_queued_cmd = 0
        self.pause_cmd_flag = 0
        self.velocity_ratio = 0
        self.acceleration_ratio = 0
        self.jerk_ratio = 0
        self.xyz_velocity_ratio = 0
        self.r_velocity_ratio = 0
        self.xyz_acceleration_ratio = 0
        self.r_acceleration_ratio = 0
        self.xyz_jerk_ratio = 0
        self.r_jerk_ratio = 0
        self.brake_status = 0
        self.robot_enable_status = 0
        self.drag_status = 0
        self.running_status = 0
        self.robot_error_state = 0
        self.jog_status = 0
        self.robot_type = 0
        self.drag_button_signal = 0
        self.enable_button_signal = 0
        self.record_button_signal = 0
        self.reappear_button_signal = 0
        self.jaw_button_signal = 0
        self.six_force_online = 0
        self.vibration_dis_z = 0.0
        self.robot_current_command_id = 0
        self.m_actual = []
        self.load = 0.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.center_z = 0.0
        self.user_value = []
        self.tool_value = []
        self.trace_index = 0.0
        self.six_force_value = []
        self.target_quaternion = []
        self.actual_quaternion = []
        self.auto_manual_mode = []


class DobotApiFeedback(DobotApiBase):
    """Interface for feedback data"""

    def __init__(self, ip: IPStr, port: PortInt, logger: logging.Logger) -> None:
        super().__init__(ip=ip, port=port, logger=logger)
        self.__lock = threading.Lock()
        self.__global_lock_value = threading.Lock()
        self.__robot_sync_break = threading.Event()
        self.__feedback_message = np.empty((0,))
        self.__feed_data = RobotStatusItem()

        threading.Thread(target=self._receive_feedback_data, daemon=True).start()
        threading.Thread(target=self._parse_feedback_data, daemon=True).start()

    def _receive_feedback_data(self) -> None:
        """
        Receive real-time feedback from the robot.
        """
        num_read = 0
        while True:
            data = bytes()
            while num_read < 1440:
                try:
                    temp = self.socket_dobot.recv(1440 - num_read)
                    if temp:
                        num_read += len(temp)
                        data += temp
                except Exception as e:
                    self._logger.warning(e)
                    self.socket_dobot = self.reconnect(self.ip, self.port)

            num_read = 0
            with self.__lock:
                self.__feedback_message = np.frombuffer(data, dtype=FeedbackMessageDType)


    def _parse_feedback_data(self) -> None:
        """
        Parse robot feedback data and update internal state.
        """
        while True:
            with self.__lock:
                feed_info = self.__feedback_message

            try:
                if feed_info and hex(feed_info["test_value"][0]) == "0x123456789abcdef":
                    with self.__global_lock_value:
                        # Update robot data
                        self._update_feed_data(feed_info)
            except Exception as e:
                self._logger.info(f'feed_info: {feed_info}')
                self._logger.warning(e)
            sleep(0.01)

    def _update_feed_data(self, feed_info: np.ndarray) -> None:
        """
        Updates the feedback data with the parsed information from feed_info.
        """
        self.__feed_data.len = feed_info["len"][0]
        self.__feed_data.digital_input_bits = feed_info["digital_input_bits"][0]
        self.__feed_data.digital_output_bits = feed_info["digital_output_bits"][0]
        self.__feed_data.robot_mode = RobotModes(feed_info["robot_mode"][0])
        self.__feed_data.timestamp = feed_info["time_stamp"][0]
        self.__feed_data.timestamp_reserve_bit = feed_info["time_stamp_reserve_bit"][0]
        self.__feed_data.test_value = feed_info["test_value"][0]
        self.__feed_data.test_value_keep_bit = feed_info["test_value_keep_bit"][0]
        self.__feed_data.speed_scaling = feed_info["speed_scaling"][0]
        self.__feed_data.linear_momentum_norm = feed_info["linear_momentum_norm"][0]
        self.__feed_data.v_main = feed_info["v_main"][0]
        self.__feed_data.v_robot = feed_info["v_robot"][0]
        self.__feed_data.i_robot = feed_info["i_robot"][0]
        self.__feed_data.i_robot_keep_bit_1 = feed_info["i_robot_keep_bit1"][0]
        self.__feed_data.i_robot_keep_bit_2 = feed_info["i_robot_keep_bit2"][0]
        self.__feed_data.tool_accelerometer_values = feed_info["tool_accelerometer_values"].tolist()[0]
        self.__feed_data.elbow_position = feed_info["elbow_position"].tolist()[0]
        self.__feed_data.elbow_velocity = feed_info["elbow_velocity"].tolist()[0]
        self.__feed_data.q_target = feed_info["q_target"].tolist()[0]
        self.__feed_data.qd_target = feed_info["qd_target"].tolist()[0]
        self.__feed_data.qdd_target = feed_info["qdd_target"].tolist()[0]
        self.__feed_data.i_target = feed_info["i_target"].tolist()[0]
        self.__feed_data.m_target = feed_info["m_target"].tolist()[0]
        self.__feed_data.q_actual = feed_info["q_actual"].tolist()[0]
        self.__feed_data.qd_actual = feed_info["qd_actual"].tolist()[0]
        self.__feed_data.i_actual = feed_info["i_actual"].tolist()[0]
        self.__feed_data.actual_tcp_force = feed_info["actual_TCP_force"].tolist()[0]
        self.__feed_data.tool_vector_actual = feed_info["tool_vector_actual"].tolist()[0]
        self.__feed_data.tcp_speed_actual = feed_info["TCP_speed_actual"].tolist()[0]
        self.__feed_data.tcp_force = feed_info["TCP_force"].tolist()[0]
        self.__feed_data.tool_vector_target = feed_info["Tool_vector_target"].tolist()[0]
        self.__feed_data.tcp_speed_target = feed_info["TCP_speed_target"].tolist()[0]
        self.__feed_data.motor_temperatures = feed_info["motor_temperatures"].tolist()[0]
        self.__feed_data.joint_modes = feed_info["joint_modes"].tolist()[0]
        self.__feed_data.v_actual = feed_info["v_actual"][0]
        self.__feed_data.hand_type = feed_info["hand_type"][0]
        self.__feed_data.user = feed_info["user"][0]
        self.__feed_data.tool = feed_info["tool"][0]
        self.__feed_data.run_queued_cmd = feed_info["run_queued_cmd"][0]
        self.__feed_data.pause_cmd_flag = feed_info["pause_cmd_flag"][0]
        self.__feed_data.velocity_ratio = feed_info["velocity_ratio"][0]
        self.__feed_data.acceleration_ratio = feed_info["acceleration_ratio"][0]
        self.__feed_data.jerk_ratio = feed_info["jerk_ratio"][0]
        self.__feed_data.xyz_velocity_ratio = feed_info["xyz_velocity_ratio"][0]
        self.__feed_data.r_velocity_ratio = feed_info["r_velocity_ratio"][0]
        self.__feed_data.xyz_acceleration_ratio = feed_info["xyz_acceleration_ratio"][0]
        self.__feed_data.r_acceleration_ratio = feed_info["r_acceleration_ratio"][0]
        self.__feed_data.xyz_jerk_ratio = feed_info["xyz_jerk_ratio"][0]
        self.__feed_data.r_jerk_ratio = feed_info["r_jerk_ratio"][0]
        self.__feed_data.brake_status = feed_info["brake_status"][0]
        self.__feed_data.robot_enable_status = feed_info["enable_status"][0]
        self.__feed_data.drag_status = feed_info["drag_status"][0]
        self.__feed_data.running_status = feed_info["running_status"][0]
        self.__feed_data.robot_error_state = feed_info["error_status"][0]
        self.__feed_data.jog_status = feed_info["jog_status"][0]
        self.__feed_data.robot_type = feed_info["robot_type"][0]
        self.__feed_data.drag_button_signal = feed_info["drag_button_signal"][0]
        self.__feed_data.enable_button_signal = feed_info["enable_button_signal"][0]
        self.__feed_data.record_button_signal = feed_info["record_button_signal"][0]
        self.__feed_data.reappear_button_signal = feed_info["reappear_button_signal"][0]
        self.__feed_data.jaw_button_signal = feed_info["jaw_button_signal"][0]
        self.__feed_data.six_force_online = feed_info["six_force_online"][0]
        self.__feed_data.vibration_dis_z = feed_info["vibrationdisZ"][0]
        self.__feed_data.robot_current_command_id = feed_info["currentcommandid"][0]
        self.__feed_data.m_actual = feed_info["m_actual"].tolist()[0]
        self.__feed_data.load = feed_info["load"][0]
        self.__feed_data.center_x = feed_info["center_x"][0]
        self.__feed_data.center_y = feed_info["center_y"][0]
        self.__feed_data.center_z = feed_info["center_z"][0]
        self.__feed_data.user_values = feed_info["user[6]"].tolist()[0]
        self.__feed_data.tool_values = feed_info["tool[6]"].tolist()[0]
        self.__feed_data.trace_index = feed_info["trace_index"][0]
        self.__feed_data.six_force_values = feed_info["six_force_value"].tolist()[0]
        self.__feed_data.target_quaternion = feed_info["target_quaternion"].tolist()[0]
        self.__feed_data.actual_quaternion = feed_info["actual_quaternion"].tolist()[0]
        self.__feed_data.auto_manual_mode = feed_info["auto_manual_mode"].tolist()[0]

    def get_feed_data(self) -> RobotStatusItem:
        """
        Return the current robot status.
        """
        with self.__global_lock_value:
            return copy(self.__feed_data)

    def wait_until_arrive(self, command_id: int) -> None:
        """
        Wait until the robot has completed the motion command with the specified command ID.
        """
        while True:
            while not self.__robot_sync_break.is_set():
                with self.__global_lock_value:
                    if self.__feed_data.robot_enable_status and self.__feed_data.robot_current_command_id > command_id:
                        break
                    elif self.__feed_data.robot_current_command_id == command_id and self.__feed_data.robot_mode == 5:
                        break
                sleep(0.01)
            self.__robot_sync_break.clear()
            break

    def exit_sync(self) -> None:
        """
        Exit synchronization mode.
        """
        self.__robot_sync_break.set()
