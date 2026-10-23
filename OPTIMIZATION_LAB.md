# OPTI: Vorkonditioniertes Gradientenverfahren

Das Notebook [Gradientenverfahren.ipynb](Gradientenverfahren.ipynb) enthält die
Erklärungen zu Algorithmus 4.11 (Armijo) und Algorithmus 4.12 (exakte
Schrittweite bei quadratischer Zielfunktion). Im zweiten Teil lassen sich
mehrere Zielfunktionen im Browser auswählen und untersuchen.

## Notebook starten

Öffne `Gradientenverfahren.ipynb` aus diesem Ordner und führe die letzte
Codezelle aus. Sie zeigt die Seite
[Interactive_Optimization_Lab.html](Interactive_Optimization_Lab.html) direkt
im Notebook an. Die Abbildungen werden von Plotly dargestellt, das beim Laden
eine Internetverbindung benötigt.

Falls die Notebook-Ansicht den eingebetteten Inhalt nicht zeigt, lässt sich die
HTML-Datei auch direkt im Browser öffnen. Die Notebook-Codezelle sucht die
Datei im aktuellen Ordner und in dessen übergeordneten Ordnern.

## Browser-Beispiel

Im Auswahlfeld **Zielfunktion** stehen die Beispiele aus
`src/optimization_examples.py` zur Verfügung:

* **Quadratische Funktion:** `f(x) = 1/2 xᵀ H x + bᵀ x + c`; die Eingabefelder
  für `H`, `b` und `c` ändern die Funktion. `H` muss positiv definit sein.
* **Himmelblau:** `f(x) = (x₁² + x₂ − 11)² + (x₁ + x₂² − 7)²`.
* **Rosenbrock:** `f(x) = (a − x₁)² + b(x₂ − x₁²)²`; die Parameter `a` und `b`
  lassen sich einstellen und müssen positiv sein.

Die Seite vergleicht die euklidische Metrik, eine eigene Metrik `M` und eine
diagonale Metrik. Beim quadratischen Beispiel ist diese der
Jacobi-Vorkonditionierer `diag(H)`; bei Himmelblau und Rosenbrock wird eine
positive Diagonalmatrix aus den Beträgen der Hessendiagonale am Startpunkt
gebildet. Die Einträge `M11`, `M12` und `M22` legen

`M = [[M11, M12], [M12, M22]]`

fest. `M12` wird auf beide Nebendiagonaleinträge übernommen; `M` muss positiv
definit sein. Beim Wechsel der Zielfunktion wird ein passender Startpunkt
gesetzt, der anschließend über `x₀` angepasst werden kann. Mit Enter oder
**Neu berechnen** werden Änderungen übernommen.

Die sechs Abbildungen zeigen Höhenlinien und Wege, die Zielfläche, die
Variablen pro Iteration, die Armijo-Suche, die verallgemeinerten Eigenwerte und
eine Kennzahlentabelle. Für Himmelblau und Rosenbrock wird das beste Ergebnis
der drei Läufe markiert; wegen möglicher lokaler Minima ist dies keine
Garantie für das globale Minimum. Spektren und Konditionszahlen werden nur für
die quadratische Funktion angezeigt. Zoomen und Drehen geht direkt mit der
Maus. Die Metrik-Auswahl und der Schieberegler zeigen den aktuellen
Iterationspunkt; **Zurück**, **Weiter** und **Start** bewegen ihn durch die
Iterationen.

## Python-Module

Die Python-Dateien in `src` bleiben als eigene Implementierungen erhalten:

* `gradient_methods.py` enthält das Gradientenverfahren mit Armijo-Suche und
  die exakte Schrittweite für quadratische Schnittfunktionen.
* `optimization_examples.py` enthält die quadratische Funktion sowie
  Himmelblau- und Rosenbrock-Beispiele mit Ableitungen und Diagrammen.
* `optimization_lab.py` enthält außerdem eine Matplotlib-Version des
  quadratischen Vergleichs.

Die Browser-Seite ruft diese Python-Module nicht direkt auf. Sie bildet die
Zielfunktionen aus `optimization_examples.py` und die Armijo-Schritte
eigenständig mit JavaScript nach, damit Eingabefelder und Diagramme im Browser
direkt reagieren. Wenn sich die Python-Funktionen ändern, müssen ihre
JavaScript-Gegenstücke in `Interactive_Optimization_Lab.html` entsprechend
aktualisiert werden. Die Algorithmuserklärungen im Notebook gelten für die
Verfahren im Python-Code; das Browser-Beispiel veranschaulicht die Verfahren
interaktiv.

Wenn die Python-Beispiele oder die Matplotlib-Version verwendet werden sollen,
installiere die Abhängigkeiten in der Notebook-Umgebung:

```powershell
python -m pip install -r requirements-lab.txt
```

## Tests

Die Tests lassen sich aus diesem Ordner starten:

```powershell
python -m unittest discover -s tests -v
```
