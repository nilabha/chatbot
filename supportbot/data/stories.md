## path 1              <!-- name of the story - just for debugging -->
* greet              
  - utter_greet
* informa               <!-- user utterance, in format _intent[entities] -->
  - utter_informa

## path 2               <!-- this is already the start of the next story -->
* greet
  - utter_greet             <!-- action of the bot to execute -->
* informb
  - utter_informb

## path 3
* greet
  - utter_greet
* informc
  - utter_informc
  
## path 4               <!-- this is already the start of the next story -->
* greet
  - utter_greet             <!-- action of the bot to execute -->
* informd
  - utter_informd

## path 5
* greet
  - utter_greet
* informe
  - utter_informe

## path 6               <!-- this is already the start of the next story -->
* greet
  - utter_greet             <!-- action of the bot to execute -->
* informf
  - utter_informf

## path 7
* greet
  - utter_greet
* informg
  - utter_informg
  
## path 8               <!-- this is already the start of the next story -->
* greet
  - utter_greet             <!-- action of the bot to execute -->
* informh
  - utter_informh

## path 9
* greet
  - utter_greet
* informi
  - utter_informi

## say goodbye
* goodbye
  - utter_goodbye
