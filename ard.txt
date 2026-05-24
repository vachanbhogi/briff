#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// Initialize LCD: address 0x27 (common), 16 columns, 2 rows
LiquidCrystal_I2C lcd(0x27, 16, 2);

// --- PIN CONFIGURATION ---
const int ledPin = 13;
const int buzzerPin = 9;

// --- CPR TEMPO CONFIGURATION ---
const float bpm = 104.0; 
const unsigned long blinkInterval = (60.0 / bpm) * 1000 / 2; 

// --- TIMING VARIABLES ---
unsigned long previousBlinkMillis = 0;
int ledState = LOW;

// --- AUDIO & STATE SYSTEM ---
char cprState = 'S'; // 'G'=Good, 'F'=Fast, 'L'=sLow, 'S'=Stop
unsigned long soundStateMillis = 0;
int noteIndex = 0;

// "STAYIN' ALIVE" MELODY ARRAYS (Lower Octave Version)
const int happyMelody[] = {
  523, 466, 415, 392,   // Ah, ha, ha, ha
  311, 350, 350, 350,   // stay-in' a - live
  311, 350, 350, 350    // stay-in' a - live
};

const int noteDurations[] = {
  606, 545, 545, 545,   
  545, 182, 121, 182,   
  606, 182, 121, 182    
};
const int totalNotes = 12; 

// "VICTORY" MELODY (Heimlich Success)
const int victoryMelody[] = {523, 659, 784, 1047}; // C5, E5, G5, C6
const int victoryDurations[] = {150, 150, 150, 400};
const int totalVictoryNotes = 4;

void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  Serial.begin(9600); 
  
  // Initialize and turn on the LCD screen
  lcd.init();
  lcd.backlight();
  
  // Display initial startup splash screen
  updateLCD("  CPR TRAINING  ", "   READY...     ");
}

void loop() {
  unsigned long currentMillis = millis();

  // 1. METRONOME (Always blinks at 104 BPM)
  if (currentMillis - previousBlinkMillis >= blinkInterval) {
    previousBlinkMillis = currentMillis;
    ledState = (ledState == LOW) ? HIGH : LOW;
    digitalWrite(ledPin, ledState);
  }

  // 2. COMMUNICATION & LCD UPDATE
  if (Serial.available() > 0) {
    char incomingChar = Serial.read();
    
    // Check if the received character is a valid state command
    if (incomingChar == 'G' || incomingChar == 'F' || incomingChar == 'L' || incomingChar == 'S' || incomingChar == 'V') {
      if (incomingChar != cprState) { 
        cprState = incomingChar;
        noteIndex = 0; 
        soundStateMillis = 0; 
        
        // Dynamically change the LCD display text based on the code received
        switch(cprState) {
          case 'G':
            updateLCD("    PERFECT!    ", " STAYIN' ALIVE! ");
            break;
          case 'F':
            updateLCD("   TOO FAST!    ", "  --SLOW DOWN-- ");
            break;
          case 'L':
            updateLCD("   TOO SLOW!    ", "  ++SPEED UP++  ");
            break;
          case 'S':
            updateLCD("  SYSTEM READY  ", " WAITING...     ");
            break;
          case 'V':
            updateLCD("  J-HOOK VALID  ", "    VICTORY!    ");
            break;
        }
      }
    }
  }

  // 3. AUDIO MACHINE
  handleAudio(currentMillis);
}

// Helper function to cleanly rewrite both rows of the LCD screen without flickering
void updateLCD(String line1, String line2) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(line1);
  lcd.setCursor(0, 1);
  lcd.print(line2);
}

void handleAudio(unsigned long currentMillis) {
  if (cprState == 'S') {
    noTone(buzzerPin);
    return;
  }

  // STATE: GOOD TEMPO (Play Stayin' Alive)
  if (cprState == 'G') {
    int currentDuration = noteDurations[noteIndex];
    if (currentMillis - soundStateMillis >= currentDuration) {
      soundStateMillis = currentMillis;
      tone(buzzerPin, happyMelody[noteIndex], currentDuration - 30);
      noteIndex = (noteIndex + 1) % totalNotes; 
    }
  }

  // STATE: VICTORY CHIME (Heimlich)
  if (cprState == 'V') {
    if (noteIndex < totalVictoryNotes) {
      int currentDuration = victoryDurations[noteIndex];
      if (currentMillis - soundStateMillis >= currentDuration) {
        soundStateMillis = currentMillis;
        tone(buzzerPin, victoryMelody[noteIndex], currentDuration - 30);
        noteIndex++; 
      }
    }
  }

  // STATE: FASTER OR SLOWER TEMPO (Both trigger the warning sound)
  if (cprState == 'F' || cprState == 'L') {
    int warningPitch = (cprState == 'F') ? 800 : 230; // High warning for fast, low warning for slow
    int alarmSpeed = 150;   
    
    if (currentMillis - soundStateMillis >= alarmSpeed) {
      soundStateMillis = currentMillis;
      if (noteIndex % 2 == 0) {
        tone(buzzerPin, warningPitch, alarmSpeed - 30);
      } else {
        noTone(buzzerPin); 
      }
      noteIndex++;
    }
  }
}