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

const int arraySize = 500; 
int NOValues[arraySize];
int RefValues[arraySize];
volatile int currentIndex = 0; // Index to keep track of the current position in the array
int differenceRef ;   // Difference between the maximum and minimum values
int differenceNO ;   // Difference between the maximum and minimum values
volatile bool readyToSend = false;



// Variables
const unsigned long pumpDuration = 15000; 
const unsigned long IRSourceDuration= 500; // Duration of air pump activation in milliseconds [5 seg]
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

char getRandomLetter() {
  int randomNumber = random(26); // Generate a random number between 0 and 25
  char randomLetter = 'a' + randomNumber; // Convert the random number to an ASCII character in the range 'a' to 'z'
  return randomLetter;
}

static void prepareTxFrame( uint8_t port )
{
	/*appData size is LORAWAN_APP_DATA_MAX_SIZE which is defined in "commissioning.h".
	*appDataSize max value is LORAWAN_APP_DATA_MAX_SIZE.
	*if enabled AT, don't modify LORAWAN_APP_DATA_MAX_SIZE, it may cause system hanging or failure.
	*if disabled AT, LORAWAN_APP_DATA_MAX_SIZE can be modified, the max value is reference to lorawan region and SF.
	*for example, if use REGION_CN470, 
	*the max value for different DR can be found in MaxPayloadOfDatarateCN470 refer to DataratesCN470 and BandwidthsCN470 in "RegionCN470.h".
	*/
    
  appDataSize = 4;
  char Refup[8];
  char Ref[8];
  char NOup[8];
  char NO[8];
  sprintf(Refup, "%ld", differenceRef / 256);
  sprintf(Ref, "%ld", differenceRef % 256);
  sprintf(NOup, "%ld", differenceNO / 256);
  sprintf(NO, "%ld", differenceNO % 256);
  // Assuming appData is declared as an array of uint8_t
  appData[0] = static_cast<uint8_t>(Refup[0]);
  appData[1] = static_cast<uint8_t>(Ref[0]);
  appData[2] = static_cast<uint8_t>(NOup[0]);
  appData[3] = static_cast<uint8_t>(NO[0]);
}


// Task handles for Core 0 and Core 1
/*
TaskHandle_t core0TaskHandle = NULL;
TaskHandle_t core1TaskHandle = NULL;

void core0Task(void* parameter);
void core1Task(void* parameter);
*/


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
  
  // Print a message to the Serial Monitor
  Serial.println("Solenoid Valve Control");

  
  //LoRa
  Mcu.begin();
  deviceState = DEVICE_STATE_INIT;

/*
xTaskCreatePinnedToCore(core0Task, "Core 0 Task", 10000, NULL, 1, &core0TaskHandle, 0);
xTaskCreatePinnedToCore(core1Task, "Core 1 Task", 10000, NULL, 1, &core1TaskHandle, 1);


  
  
// Configure Timer0 to generate a 10 Hz pulse square wave for the IR source
noInterrupts();

// Set Timer1 to 1 Hz (1 second period)
Timer1.attachInterrupt(timer1ISR);
Timer1.initialize(1000000); // 1 second = 1,000,000 microseconds
Timer1.stop(); // Stop the timer initially
// Configure LEDC to generate a 10Hz pulse with 50% duty cycle on irSourcePin
ledcSetup(0, 10, 8);
ledcAttachPin(irSourcePin, 0);

// Configure Timer2 to trigger an interrupt at approximately 0.017 Hz (1 time every 60 seconds) for LoRaWAN transmission
Timer2.attachInterrupt(timer2ISR);
Timer2.initialize(58593750); // 1 minute = 58,593,750 microseconds
Timer2.stop(); // Stop the timer initially
interrupts();

*/
}

void loop() {
  
    //activateAirFlow();
    
    //delay(10000);

    IRPulse();

    Serial.print("REF Amplitude: ");
    differenceRef = computeDifference(RefValues);
    Serial.println(differenceNO);
    Serial.print("NO Amplitude: ");
    differenceNO = computeDifference(NOValues);
    Serial.println(differenceNO);
    


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

  NOValues[currentIndex] = readingNO; 
  RefValues[currentIndex] = readingRef;

  currentIndex=(currentIndex + 1) % arraySize;   // Store the sensor value in the array
  //Serial.println(reading);
  // Move to the next index or wrap around if we reached the end of the array
  delay(1);
}


void IRPulse(){
  delay(500);
  previousMillis = millis();
  // Print a message to the Serial Monitor
  Serial.println("ir source  activated");
  ledcWrite(ledChannel, dutyCycle); 
  // Wait for the specified duration
  while (millis() - previousMillis < IRSourceDuration) {
    
    //ledcWrite(ledChannel, dutyCycle); 
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






