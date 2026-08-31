function [device,genCLI] = init_stage(home,type)
%% init lib and connect

if nargin<1
    home = false;
end
if nargin<2
    type = 1;
end

if type ==1
    % for rotation stage
    serialNumber = '27266129';
    DeviceSettingsName = 'PRM1-Z8';
elseif type ==2
    % for z stage
    serialNumber = '27266098';
    DeviceSettingsName = 'Z825B';
end

timeout_val=60000;

% load lib
devCLI = NET.addAssembly('C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI.dll');
genCLI = NET.addAssembly('C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.GenericMotorCLI.dll');
motCLI = NET.addAssembly('C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.KCube.DCServoCLI.dll');

import Thorlabs.MotionControl.DeviceManagerCLI.*
import Thorlabs.MotionControl.GenericMotorCLI.*
import Thorlabs.MotionControl.KCube.DCServoCLI.*

% Build Device List loads the connected devices to available memory
DeviceManagerCLI.BuildDeviceList();

% Connect to the controller
device = KCubeDCServo.CreateKCubeDCServo(serialNumber);
device.Connect(serialNumber);

try
    % Try/Catch statement used to disconnect correctly after an error

    device.WaitForSettingsInitialized(5000);
    device.StartPolling(250);
    
    %Pull the enumeration values from the DeviceManagerCLI
    optionTypeHandle = devCLI.AssemblyHandle.GetType('Thorlabs.MotionControl.DeviceManagerCLI.DeviceSettingsSectionBase+SettingsUseOptionType');
    optionTypeEnums = optionTypeHandle.GetEnumValues(); 
    
    %Load Settings to the controller
    motorConfiguration = device.LoadMotorConfiguration(serialNumber);
    motorConfiguration.LoadSettingsOption = optionTypeEnums.Get(1); % File Settings Option
    motorConfiguration.DeviceSettingsName = DeviceSettingsName; %The actuator type needs to be set here. This specifically loads an PRM1-Z8
    factory = KCubeMotor.KCubeDCMotorSettingsFactory();
    device.SetSettings(factory.GetSettings(motorConfiguration), true, false);
    
    % Enable the device and start sending commands
    device.EnableDevice();
    pause(1.5); %wait to make sure the cube is enabled
    disp('Stage connected!')
    % Home the stage
    if home 
        msgbox("Stage Homing...");
        device.Home(timeout_val);
        msgbox("Stage Homed!",'modal')
    end

    % show pos
    pos = System.Decimal.ToDouble(device.Position);
    msgbox(["Connected!"; sprintf("The stage motor position is %.4f now!",pos)],'modal');
catch e
    msgbox("Error has caused the program to stop, disconnecting..")
    msgbox([string(e.identifier);string(e.message)],'modal');
    device.StopPolling();
    device.Disconnect();
end
