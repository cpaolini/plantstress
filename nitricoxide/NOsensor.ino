#include <heltec.h>
#include "LoRaWan_APP.h"
#include <SPI.h>


// Pin Definitions
const int batteryPin = 1;    // Pin connected to the Battery Capacity sensor
const int NOgasPin = 3;      // Pin connected to the NO gas output sensor
const int REFgasPin = 2;     // Pin connected to the Reference output sensor
const int IRSourcePin = 4;   // Pin connected to the IR Source
const int airPumpPin = 6;    // Pin connected to the air pump
const int solenoidPin = 7;   // Pin connected to the solenoid valve

//Ir source control
const int freq = 10;        // 10Hz frequency
const int ledChannel = 0;   // Channel 0
const int resolution = 12;  // 12 bit resolution
const int dutyCycle = 2048; // 25% Duty Cycle

const int arraySize = 4000; 
int NOValues[arraySize];
int RefValues[arraySize];
volatile int currentIndex = 0; // Index to keep track of the current position in the array
int differenceNO ;
int differenceRef ;   // Difference between the maximum and minimum values
volatile bool readyToSend = false;



// Variables
const unsigned long pumpDuration = 15000; 
const unsigned long IRSourceDuration= 4000; // Duration of ir pulse  in milliseconds [8 seg]
unsigned long previousMillis = 0;       // Stores the previous timestamp for comparison
float amplitudeRef; // Peak to peak value of the Reference
float amplitudeNo; // Peak to peak value of the Nitric Oxide Gas
float readingNO=1; //reading of the sensor
float readingRef=1;

//LoRa

/* OTAA para*/
uint8_t devEui[] = { 0x22, 0x32, 0x33, 0x00, 0x00, 0x88, 0x88, 0x02 };
uint8_t appEui[] = { 0x22, 0x32, 0x33, 0x00, 0x00, 0x88, 0x88, 0x02 };
uint8_t appKey[] =  { 0x22, 0x32, 0x33, 0x00, 0x00, 0x88, 0x88, 0x02};

/* ABP para*/
uint8_t nwkSKey[] = { 0xde, 0xb1, 0xda, 0xd2, 0xfa, 0xd3, 0xba, 0xd4, 0xfa, 0xb5, 0xfe, 0xd6, 0xab, 0xba, 0xdb,0xba };
uint8_t appSKey[] = { 0xde, 0xb1, 0xda, 0xd2, 0xfa, 0xd3, 0xba, 0xd4, 0xfa, 0xb5, 0xfe, 0xd6, 0xab, 0xba, 0xdb,0xba };
uint32_t devAddr =  ( uint32_t )0x007e6ae1;

/*LoraWan channelsmask, default channels 0-7*/ 
uint16_t userChannelsMask[6]={ 0x00FF,0x0000,0x0000,0x0000,0x0000,0x0000 };

/*LoraWan region, select in arduino IDE tools*/
LoRaMacRegion_t loraWanRegion = ACTIVE_REGION;

/*LoraWan Class, Class A and Class C are supported*/
DeviceClass_t  loraWanClass = CLASS_A;

/*the application data transmission duty cycle.  value in [ms].*/
uint32_t appTxDutyCycle = 15000;

/*OTAA or ABP*/
bool overTheAirActivation = false;

/*ADR enable*/
bool loraWanAdr = true;

/* Indicates if the node is sending confirmed or unconfirmed messages */
bool isTxConfirmed = true;

/* Application port */
uint8_t appPort = 2;

/* Number of trials to transmit the frame */
uint8_t confirmedNbTrials = 4;



static void prepareTxFrame( uint8_t port )
{
	/*appData size is LORAWAN_APP_DATA_MAX_SIZE which is defined in "commissioning.h".
	*appDataSize max value is LORAWAN_APP_DATA_MAX_SIZE.
	*if enabled AT, don't modify LORAWAN_APP_DATA_MAX_SIZE, it may cause system hanging or failure.
	*if disabled AT, LORAWAN_APP_DATA_MAX_SIZE can be modified, the max value is reference to lorawan region and SF.
	*for example, if use REGION_CN470, 
	*the max value for different DR can be found in MaxPayloadOfDatarateCN470 refer to DataratesCN470 and BandwidthsCN470 in "RegionCN470.h".
	*/
  

  char buf[16];

  sprintf(buf, "%ld,%ld,%ld,%ld", differenceRef / 256, differenceRef % 256, differenceNO / 256, differenceNO % 256);

  int bufLength=strlen(buf);
  appDataSize = bufLength;
  
  for (int i = 0; i < bufLength; i++) {
    appData[i] = buf[i];
} 

  }



void setup() {

  Heltec.begin(true /*DisplayEnable Enable*/, true /*LoRa Disable*/, true /*Serial Enable*/, true /*PABOOST Enable*/, 470E6 /**/);


  // Initialize the pins as inputs outputs

  pinMode(batteryPin, INPUT);
  pinMode(NOgasPin, INPUT);
  pinMode(REFgasPin, INPUT);

  pinMode(solenoidPin, OUTPUT);
  pinMode(airPumpPin, OUTPUT);
  pinMode(IRSourcePin, OUTPUT);
  
  

  // Set initial states
  digitalWrite(solenoidPin, LOW);  // Turn off the solenoid valve
  digitalWrite(airPumpPin, LOW);   // Turn off the air pump
  digitalWrite(IRSourcePin, LOW);  // Turn off the solenoid valve
  

  // configure LED PWM functionalitites
  ledcSetup(ledChannel, freq, resolution);
  
  // attach the channel to the pin to be controlled
  ledcAttachPin(IRSourcePin, ledChannel);
  
  // Set up Serial communication at a baud rate of 115200
  Serial.begin(115200);
 

  //LoRa
  Mcu.begin();
  deviceState = DEVICE_STATE_INIT;

}


void loop() {
  
    
    activateAirFlow();
    
    delay(5000);

    IRPulse();

    //Serial.print("REF Amplitude: ");
    differenceRef = computeDifference(RefValues);
    Serial.print(differenceRef);
    //Serial.print("NO Amplitude: ");
    Serial.print(", ");
    differenceNO = computeDifference(NOValues);
    Serial.print(differenceNO);
    Serial.print(", ");
    
  
    LoRasend();
    
}



void activateAirFlow() {
  // Activate the solenoid valve
  digitalWrite(solenoidPin, HIGH);  // Turn on the solenoid valve
  delay(500);
  // Store the current timestamp
  previousMillis = millis();
  
  // Print a message to the Serial Monitor
  Serial.println("Solenoid valves activated");
  
  // Wait for the specified duration
  while (millis() - previousMillis < pumpDuration) {
    // Activate the air pump
    digitalWrite(airPumpPin, HIGH);  // Turn on the air pump
  
  }
  // Deactivate the air pump
  digitalWrite(airPumpPin, LOW);  // Turn off the air pump
  
  delay(500);
  // Deactivate the solenoid valve
  digitalWrite(solenoidPin, LOW);  // Turn off the solenoid valve

  delay(10000);
  
  // Print a message to the Serial Monitor
  Serial.println("Solenoid valves deactivated");
}


void readSensor(){

  readingNO=analogRead(NOgasPin);
  readingRef=analogRead(REFgasPin);

 // Store the sensor value in the array
  NOValues[currentIndex] = readingNO; 
  RefValues[currentIndex] = readingRef;

  currentIndex=(currentIndex + 1) % arraySize;   
  
  delay(1);
}


void IRPulse(){
  delay(500);
  previousMillis = millis();
  // Print a message to the Serial Monitor
  //Serial.println("ir source  activated");
  ledcWrite(ledChannel, dutyCycle); 
  // Wait for the specified duration
  while (millis() - previousMillis < IRSourceDuration) {
    readSensor();
  }

  ledcWrite(ledChannel, 0);
  
}


void LoRasend(){
  switch( deviceState )
  {
    case DEVICE_STATE_INIT:
    {
#if(LORAWAN_DEVEUI_AUTO)
      LoRaWAN.generateDeveuiByChipID();
#endif
      LoRaWAN.init(loraWanClass,loraWanRegion);
      break;
    }
    case DEVICE_STATE_JOIN:
    {
      LoRaWAN.join();
      break;
    }
    case DEVICE_STATE_SEND:
    {
      prepareTxFrame( appPort );
      
      LoRaWAN.send();
      deviceState = DEVICE_STATE_CYCLE;
      break;
    }
    case DEVICE_STATE_CYCLE:
    {
      // Schedule next packet transmission
      LoRaWAN.cycle(appTxDutyCycle);
      deviceState = DEVICE_STATE_SLEEP;
      break;
    }
    case DEVICE_STATE_SLEEP:
    {
      LoRaWAN.sleep(loraWanClass);
      break;
    }
    default:
    {
      deviceState = DEVICE_STATE_INIT;
      break;
    }
  
  }
  delay(120000);
}

int computeDifference(int adcValues[]) {
  // Find the minimum and maximum values in the adcValues array
  int minValue = adcValues[0];
  int maxValue = adcValues[0];

  for (int i = 1; i < arraySize; i++) {
    if (adcValues[i] < minValue) {
      minValue = adcValues[i];
    }
    if (adcValues[i] > maxValue) {
      maxValue = adcValues[i];
    }
  }

  // Calculate and return the difference between the maximum and minimum values
  int difference = maxValue - minValue;
  return difference;
}






