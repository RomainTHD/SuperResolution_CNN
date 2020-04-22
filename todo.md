Pour override la fonction de target : data.py/target_transform ? ou dataset.py/load_img ?

Comment ça marche :
  * prend une image
  * la réduit par n
  * agrandit cette image réduite par n via CNN
  * compare les 2

-> perte de qualité

Tester un truc cool :
  * prend une image
  * utilise un algo déjà existant pour agrandir
  * agrandit l'originale par n
  * compare

Comparer et être critique sur le CNN -> il imite bien l'algo géométrique ou non
problème : nécessite bien plus de puissance et nécessite un autre algo


Changer ça justement pour algos custom (genre pixel art)


Essayer avec d'autres tailles d'input


CNN : invente des données donc c'est pour ça qu'une couleur unie est zarb
c'est le but du CNN, et du coup il se plante quand les données sont linéaires parce que lui il est pas linéaire
à cause justement des faux minimas et maximas (cf. phénomène de Gibbs)


Essayer de créer un CNN qui va dire si telle image est pixel art ou non ?


Problème de couleurs
