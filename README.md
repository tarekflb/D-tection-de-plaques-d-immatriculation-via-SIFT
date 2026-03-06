# Reconnaissance de plaques d’immatriculation avec SIFT

## Description du projet

Ce projet implémente un système de reconnaissance de plaques d’immatriculation à partir d’images en utilisant l’algorithme **SIFT (Scale-Invariant Feature Transform)** et la méthode **RANSAC** pour identifier les correspondances entre une image requête et une base de plaques.

L’objectif est de comparer une image de plaque d’immatriculation avec un ensemble d’images de référence afin de trouver la correspondance la plus probable en utilisant la détection et la correspondance de points clés.


## Technologies utilisées

- Python  
- OpenCV  

- SIFT (détection de points clés)  
- RANSAC (filtrage des correspondances)  
- NumPy  


## Fonctionnement de l’algorithme

Le programme suit les étapes suivantes :

1. Chargement des images de la base de données.
2. Détection des points clés sur les images avec l’algorithme SIFT.
3. Extraction des descripteurs associés aux points clés.
4. Comparaison des descripteurs entre l’image requête et les images de la base.
5. Filtrage des correspondances avec le test de Lowe.
6. Utilisation de **RANSAC** pour éliminer les correspondances erronées.
7. Identification de la plaque la plus similaire.
Le script compare les images de test avec la base de plaques et affiche la correspondance la plus probable.




****Voici une simulation réelle d'un jeu de dataset d'une entreprise allemande, avec les vehecule de ses employés :****



<img width="1130" height="1014" alt="Capture d&#39;écran 2026-03-06 231516" src="https://github.com/user-attachments/assets/46cd28f0-bed5-40cd-a523-541fa6f9bb89" />








## Objectif pédagogique

Ce projet permet de mettre en pratique plusieurs concepts importants :

- vision par ordinateur  
- détection de caractéristiques locales  
- correspondance d’images  
- filtrage de correspondances avec RANSAC  
- traitement d’images avec OpenCV  

## Auteur

Tarek



