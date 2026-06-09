# Proiect Machine Learning — Clasificare vertebrală pe măsurători biomecanice

## 1. Problema

Pe baza unor măsurători biomecanice ale coloanei și pelvisului, modelul trebuie să decidă dacă un pacient este `Normal` sau `Abnormal`, unde `Abnormal` reprezintă apartenența la o categorie de afecțiune vertebrală.

## 2. Dataset

Datasetul are 310 observații și 7 coloane: 6 feature-uri numerice și o coloană țintă.

Feature-uri:

- pelvic incidence
- pelvic tilt
- lumbar lordosis angle
- sacral slope
- pelvic radius
- grade of spondylolisthesis

Target: `Class_att`, cu clasele `Normal` și `Abnormal`.

Distribuția claselor:

- Abnormal: 210 exemple, 67.74%
- Normal: 100 exemple, 32.26%

Concluzie: datasetul este moderat dezechilibrat, deoarece clasa `Abnormal` este de aproximativ două ori mai mare decât clasa `Normal`.

## 3. Verificarea datelor

Nu există valori lipsă în dataset. Toate feature-urile sunt numerice de tip float, deci nu este necesar encoding pentru variabile categoriale.

Au fost identificați outlieri prin regula IQR. Cei mai observați outlieri apar la:

- pelvic tilt: 13 valori extreme
- pelvic radius: 11 valori extreme
- grade of spondylolisthesis: 10 valori extreme

Din acest motiv, scalarea este importantă pentru modele sensibile la distanțe sau gradient, precum kNN și MLP. Pentru arborele de decizie, scalarea nu schimbă deciziile în mod semnificativ.

## 4. Reproductibilitate

Setări folosite:

- seed = 42
- target = Class_att
- split principal = 80% train / 20% test
- stratify = y
- clasa pozitivă = Abnormal

## 5. Separarea corectă a etapelor

Preprocesarea corectă se face după split: scalerul este antrenat doar pe train și apoi aplicat pe test. Varianta greșită este să aplicăm scalerul pe întreg datasetul înainte de split, deoarece testul influențează indirect transformarea datelor. Aceasta este data leakage.

## 6. Modele folosite

Au fost evaluate patru modele obligatorii:

- kNN
- Decision Tree
- Gaussian Naive Bayes
- MLP / ANN

## 7. Rezultate finale pe split 80/20

| Model         | Accuracy | F1 macro | ROC AUC |
| ------------- | -------: | -------: | ------: |
| kNN           |   0.8548 |   0.8267 |  0.8655 |
| Decision Tree |   0.8871 |   0.8725 |  0.9369 |
| Gaussian NB   |   0.7903 |   0.7773 |  0.8786 |
| MLP           |   0.8871 |   0.8652 |  0.9560 |

Concluzie: Decision Tree și MLP au avut cele mai bune rezultate pe splitul principal. Decision Tree are avantajul interpretabilității, iar MLP are AUC foarte bun, dar este mai greu de explicat.

## 8. Matrice de confuzie

Pentru Decision Tree:

|               | Pred Normal | Pred Abnormal |
| ------------- | ----------: | ------------: |
| True Normal   |          17 |             3 |
| True Abnormal |           4 |            38 |

Interpretare:

- TN = 17 pacienți normali clasificați corect
- FP = 3 pacienți normali clasificați greșit ca abnormal
- FN = 4 pacienți abnormal clasificați greșit ca normal
- TP = 38 pacienți abnormal clasificați corect

## 9. Cross-validation

Rezultate F1 macro:

| Model         |          5-fold mean | 5-fold std |         10-fold mean | 10-fold std |
| ------------- | -------------------: | ---------: | -------------------: | ----------: |
| kNN           |               0.7790 |     0.0481 |               0.7856 |      0.0700 |
| Decision Tree |               0.7781 |     0.0472 |               0.7789 |      0.0547 |
| Gaussian NB   |               0.7700 |     0.0211 |               0.7626 |      0.0696 |
| MLP           | aproximativ variabil |   mai mare | aproximativ variabil |    mai mare |

Concluzie: kNN, Decision Tree și GaussianNB sunt relativ stabile. MLP este mai sensibil la split și la inițializare.

## 10. Scalare

kNN este sensibil la scalare deoarece folosește distanțe între puncte. Decision Tree nu este sensibil la scalare, deoarece folosește praguri pe fiecare feature.

În experimente, kNN a avut rezultate diferite cu StandardScaler, MinMaxScaler și RobustScaler. Pentru Decision Tree rezultatele au rămas practic identice.

## 11. Resampling

Pentru că datasetul este moderat dezechilibrat, au fost testate scenarii fără resampling, cu RandomOverSampler și cu SMOTE. În aceste experimente, resampling-ul nu a îmbunătățit scorul global pe test. Totuși, resampling-ul poate fi util când obiectivul principal este creșterea recall-ului pe clasa minoritară.

## 12. Studiu parametru

Pentru kNN au fost testate valori diferite pentru `n_neighbors`: 1, 3, 5, 7, 9, 11, 15.

Cea mai bună valoare pe splitul principal a fost în jur de k=5. La k foarte mic modelul devine sensibil la zgomot. La k mai mare modelul devine mai stabil, dar poate netezi prea mult frontiera de decizie.

Pentru Decision Tree, cea mai bună adâncime a fost `max_depth=4`. Arborii prea mici au underfitting, iar arborii prea adânci pot învăța detalii specifice setului de train.

## 13. GridSearchCV

GridSearchCV a fost aplicat pentru două modele:

kNN:

- metric: manhattan
- n_neighbors: 5
- scaler: RobustScaler
- best CV F1 macro: 0.8110

Decision Tree:

- criterion: entropy
- max_depth: 5
- min_samples_leaf: 5
- best CV F1 macro: 0.8238

## 14. Interpretarea modelelor

### kNN

kNN clasifică un pacient analizând cei mai apropiați k vecini din train. Dacă majoritatea vecinilor sunt `Abnormal`, pacientul este clasificat ca `Abnormal`. Modelul este sensibil la scalare și la alegerea metricii de distanță.

Limitare: dacă feature-urile sunt pe scale diferite, distanța este dominată de variabilele cu valori mari.

### Decision Tree

Arborele ia decizii prin reguli de tipul `feature <= prag`. Cel mai important feature a fost `grade_of_spondylolisthesis`, cu importanță aproximativ 0.58. Alte feature-uri importante au fost `pelvic_radius` și `pelvic_tilt`.

Limitare: dacă arborele este prea adânc, poate face overfitting.

### Gaussian Naive Bayes

GaussianNB presupune că feature-urile sunt independente condiționat de clasă și că fiecare feature urmează aproximativ o distribuție gaussiană în interiorul fiecărei clase.

Limitare: în date biomecanice, feature-urile pot fi corelate, deci ipoteza de independență nu este complet realistă.

### MLP

MLP învață o frontieră de decizie neliniară folosind straturi ascunse. A fost folosită o arhitectură simplă, `hidden_layer_sizes=(30,)`, cu regularizare `alpha=0.001`.

Limitare: are nevoie de scalare și poate fi mai instabil decât modelele clasice pe dataseturi mici.

## 15. Analiza erorilor

Erorile apar mai ales la pacienți cu valori apropiate de zona de graniță dintre normal și abnormal. Exemplele `Normal` clasificate ca `Abnormal` au uneori valori mari la `pelvic incidence`, `pelvic tilt` sau `sacral slope`, ceea ce le face asemănătoare cu exemple patologice.

## 16. Concluzie finală

Pentru acest dataset, Decision Tree este o alegere foarte bună deoarece oferă performanță ridicată și interpretabilitate. MLP obține performanță competitivă și AUC foarte bun, dar este mai greu de justificat clinic. kNN funcționează decent, însă este sensibil la scalare. GaussianNB este simplu și stabil, dar este limitat de ipoteza de independență a feature-urilor.
