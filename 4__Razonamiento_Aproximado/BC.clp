(deftemplate Esfuerzo
   0 10 ; Define universo discurso
   (
      ; Bajo  → (0,0,2,4)  
      (bajo   (0 1) (2 1) (4 0))

      ; Medio → (2,5,8)    
      (medio  (2 0) (5 1) (8 0))

      ; Alto  → (6,8,10,10)
      (alto   (6 0) (8 1) (10 1))
   )
)

(deftemplate Horizonte
  1 10                 ; universo [1,10]
  (
    ; Corto → (1, 1, 3, 4)
    (corto  (1 1) (3 1) (4 0))

    ; Medio → (4, 5.5, 7)
    (medio  (4 0) (5.5 1) (7 0))

    ; Largo → (7, 8.5, 10, 10)
    (largo  (7 0) (8.5 1) (10 1))
  )
)

(deftemplate Beneficios
  0 10
  (
    (bajos        (1 1) (2 1) (4 0))     ; (0,0,2,4)
    (equilibrados (3 0) (5 1) (7 0))     ; (3,5,7)
    (altos        (6 0) (8 1) (9.5 0))   ; (6,8,9.5)
    (excesivos    (8 0) (9 1) (10 1))    ; (8,9,10,10)
  )
)

(deftemplate Aceptacion
  0 10
  (
    (mal-visto  (0 1) (2 1) (4 0))       ; (0,0,2,4)
    (neutral    (3 0) (5 1) (7 0))       ; (3,5,7)
    (bien-visto (6 0) (8 1) (10 1))      ; (6,8,10,10)
  )
)

(deftemplate Accion
  0 10
  (
    (descartar   (0 1) (0 1) (5 0))
    (reflexionar (2 0) (5 1) (8 0))
    (iniciar     (5 0) (10 1) (10 1))
  )
)



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; BC.clp — Base de Conocimiento (Mamdani, min–max)
;; Reglas R1..R5 tal como en la memoria
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; R1: Esfuerzo alto ∧ Horizonte largo  ⇒  Accion iniciar
(defrule R1
   (Esfuerzo alto)
   (Horizonte largo)
   =>
   (assert (Accion iniciar))
)

;; R2: Esfuerzo bajo ∧ Horizonte corto  ⇒  Accion descartar
(defrule R2
   (Esfuerzo bajo)
   (Horizonte corto)
   =>
   (assert (Accion descartar))
)

;; R3: Aceptacion mal-visto ∨ Beneficios bajos ⇒ Accion descartar
(defrule R3
   (or (Aceptacion mal-visto)
       (Beneficios bajos))
   =>
   (assert (Accion descartar))
)

;; R4: Beneficios altos ∧ Aceptacion neutral ⇒ Accion reflexionar
(defrule R4
   (Beneficios altos)
   (Aceptacion neutral)
   =>
   (assert (Accion reflexionar))
)

;; R5: Beneficios equilibrados ∧ Aceptacion bien-visto ⇒ Accion reflexionar
(defrule R5
   (Beneficios equilibrados)
   (Aceptacion bien-visto)
   =>
   (assert (Accion reflexionar))
)

(defrule final_result
   ?f <- (Accion ?A)              ; ?A es el fuzzy-value de salida
=>
   (bind ?c (moment-defuzzify ?A))    ; centroide (número)
   (printout t "Accion (centroide) = " ?c crlf)
   (if (<= ?c 3.5) then
        (printout t "Accion recomendada: DESCARTAR" crlf)
     else
        (if (>= ?c 6.5) then
             (printout t "Accion recomendada: INICIAR" crlf)
          else
             (printout t "Accion recomendada: REFLEXIONAR" crlf))))

