# Webauftritt

Im Rahmen der Entwicklung innovativer digitaler Lösungen wurde eine benutzerfreundliche Webanwendung zur Prognose von Immobilienpreisen konzipiert. Ziel dieser Anwendung ist es, sowohl privaten Käufern als auch Investoren einen einfachen Zugang zu präzisen Preisvorhersagen zu ermöglichen. Der Fokus liegt auf einer intuitiven Bedienbarkeit, die keine speziellen Fachkenntnisse voraussetzt.

![Webserver](images/webserver.png)

## Voraussetzungen

Für den Betrieb der Webanwendung sind folgende technische Voraussetzungen notwendig:

1. **Python 3.x** muss auf dem System installiert sein.
2. Installation der erforderlichen Python-Bibliotheken:

   ```bash
   pip install -r requirements.txt
   ```

3. Bereitstellung der folgenden Dateien:
   - `home.html`: Benutzeroberfläche für die Dateneingabe.
   - `zip_to_lat_lon(zip_code)`: Funktion zur Umwandlung der Postleitzahl in geografische Koordinaten.
   - `preprocess_and_predict(input_data)`: Verantwortlich für die Datenverarbeitung und Vorhersage.
   - Flask-Routen: Die Hauptseite der Anwendung wird über die Route `/` gesteuert.

## Bedienung der Startseite

Nach dem Öffnen der Anwendung im Browser erscheint die Startseite mit einem Eingabeformular. Dort können Nutzer folgende Daten eingeben:

- **Postleitzahl** (Postleitzahl des Standorts)
- **Wohnfläche** (Wohnfläche in Quadratmetern)
- **Stockwerk** (Etage des Objekts)
- **Nutzfläche** (Nutzfläche in Quadratmetern)
- **Anzahl der Stockwerke** (Gesamtanzahl der Etagen im Gebäude)
- **Grundstücksfläche** (Grundstücksgrösse in Quadratmetern)
- **Anzahl der Zimmer** (Zimmeranzahl der Immobilie)
- **Gebäude Typ** (z.B. Penthouse, Villa, Loft, etc.)

## Installation

1. **Ordner herunterladen oder navigieren:**  
   Laden Sie den Ordner `Webserver` herunter oder navigieren Sie im Terminal in das entsprechende Verzeichnis.

2. **Python-Bibliotheken:**  
   Siehe Abschnitt [Voraussetzungen](#voraussetzungen).

3. **Applikation starten:**  
   Starten Sie die Anwendung mit folgendem Befehl im Terminal:

   ```bash
   python app.py
   ```

4. **Webseite öffnen:**  
   Öffnen Sie die angegebene URL oder rufen Sie die Anwendung im Browser unter [http://localhost:5000](http://localhost:5000) auf.
