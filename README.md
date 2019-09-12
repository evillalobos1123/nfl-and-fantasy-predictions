# NFL Game Predictions

## Objetivo:

El objetivo de este proyecto es analizar las estadísticas de los jugadores de la NFL por partido para predecir su rendimiento en un partido futuro y con esto el resultado del partido.


## Intro NFL Pred:

El análisis y las predicciones se hacen tomando en cuenta no sólo las estadísticas de rendimiento sino las variables del partido que afectan a dicho rendimiento, por ejemplo, si se juega de local o visitante, el dia de la semana (ya que los lunes y jueves son partidos más importantes) y hasta el mes en el que se juega, pues la temperatura tiene gran influencia.

Tomando en cuenta estas estadisticas podemos predecir el rendimiento de un jugador tomando como variable a predecir los puntos ofensivos que realizó el equipo con dicho jugador. Con esto, podemos determinar quién es el ganador más probable de un partido futuro. 

Otra ventaja de tomar los puntos como variable a predecir es que podemos usar el resultado como guía para otros juegos como el fantasy football que mide el rendimiento individual en cada partido o hasta para tener una idea de la diferencia de puntos entre dos equipos y así conocer más a fondo el 'spread' de las apuestas.

## Pasos:

· Lo primero fue encontrar datos de los jugadores por partido ya que la mayorá vienen por temporada, pero era necesario tenerlo por partido para que se conocieran las variables del encuentro.

· Teniendo esto lo siguiente fue limpiar dichas estadisticas con un diseño de base de datos que me permitiera ver que features eran las mas significativas para la variable a predecir y cuales no nos servían.

· El paso anterior se tuvo que repetir 3 veces, una para la posicion de QB, otra para RB y una última para WR, ya que cada posición tiene diferentes estadísticas.

· Se entrenó un modelo de regresión lineal con todas las estadisticas de la base de datos ya limpia y con sólo valores numéricos.

· Se realizó una función para extraer los datos de un nuevo partido y unirlo con las estadisticas de entrenamiento del modelo para predecir los puntos por posición del juego.

· El resultado de la posición de QB son los más atinados, pues es la posición más importante asi que se le dió un peso mayor a la hora de calcular los resultados

· Se juntaron todos los pasos en una función y luego se realizó otra para hacer un DataFrame de los resultados.

## Conclusión:

El modelo realizado tiene varias áreas de oportunidad en las que trabajaré en un futuro, pero en conclusión, el resultado de el modelo actual me parecen muy favorables despues varias pruebas. Algunos partidos de la NFL son muy cerrados y por lo tanto dificiles de predecir, pero cuando la predicción del modelo te da de resultado una diferencia de puntos significativa entre dos equipos podemos estar casi seguros que sera un resultado acertado.
