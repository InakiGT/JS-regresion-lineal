let stopTraining = false;

async function getData() {
    const datosCasasR = await fetch("datos.json");
    const datosCasas = await datosCasasR.json();
    let datosLimpios = datosCasas.map( casa => ({ precio: casa.Precio, cuartos: casa.NumeroDeCuartosPromedio }) );
    datosLimpios = datosLimpios.filter(casa => (casa.precio != null && casa.cuartos != null));

    return datosLimpios;
}

function visualizarDatos(data) {
    const valores = data.map(d => ({ x: d.cuartos, y: d.precio }));
    tfvis.render.scatterplot(
        { name: 'Cuartos vs Precio' },
        { values: valores },
        {
            xLabel: 'Cuartos',
            yLabel: 'Precios',
            height: 300,
        }
    )
}

function crearModelo() {
    const modelo = tf.sequential();

    // Agregar capas
    modelo.add(tf.layers.dense({
        inputShape: [1],
        units: 1, // Neuronas
        useBias: true,
    }));

    modelo.add(tf.layers.dense({
        units: 1,
        useBias: true
    }));

    return modelo;
}

const optimizador = tf.train.adam();
const funcionPerdida = tf.losses.meanSquaredError;
const metricas = [ 'mse' ]; // Mean Squared Error

async function entrenarModelo(model, inputs, labels) {
    model.compile({
        optimizer: optimizador,
        loss: funcionPerdida,
        metrics: metricas,
    });

    const surface = { name: 'show.history live', tab: 'Training' };
    const tamanioBatch = 28; // Registros
    const epochs = 50; // Épocas
    const history = []; // Mantiene las métricas a lo largo del proceso de entrenamiento

    return await model.fit(inputs, labels, {
        tamanioBatch,
        epochs,
        suffle: true,
        callbacks: {
            onEpochEnd: ( epoch, log ) => {
                history.push(log);
                tfvis.show.history(surface, history, [ 'loss', 'mse' ]);

                if( stopTraining ) {
                    model.stopTraining = true;
                }
            }
        }
    });
}

function convertirDatosATensores(data) {
    // Se deshace de todos aquellos tensores que no sean útiles
    return tf.tidy(() => {

        // Mezcla los datos de manera aleatoria
        tf.util.shuffle(data); 
        
        const entradas = data.map(d => d.cuartos);
        const etiquetas = data.map(d => d.precio);
        // Trasforma las entradas a tensores 2D
        const tensorEntradas = tf.tensor2d(entradas, [ entradas.length, 1 ]);
        const tensorEtiquetas = tf.tensor2d(etiquetas, [ etiquetas.length, 1 ]);
        
        const entradasMax = tensorEntradas.max();
        const entradasMin = tensorEntradas.min();
        const etiquetasMax = tensorEtiquetas.max();
        const etiquetasMin = tensorEtiquetas.min();

        // Entradas normalizadas: (dato-min) / (max-min)
        const entradasNormalizadas = tensorEntradas.sub(entradasMin).div(entradasMax.sub(entradasMin));
        const etiquetasNormalizadas = tensorEtiquetas.sub(etiquetasMin).div(etiquetasMax.sub(etiquetasMin));

        return {
            entradas: entradasNormalizadas,
            etiquetas: etiquetasNormalizadas,
            entradasMax,
            entradasMin,
            etiquetasMax,
            etiquetasMin
        }

    });
}

async function verCurvaInferencia() {
    const data = await getData();
    const tensorData = await convertirDatosATensores(data);
    const { entradasMax, entradasMin, etiquetasMin, etiquetasMax } = tensorData;

    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100);
        const preds = modelo.predict(xs.reshape([ 100, 1 ]));

        const desnormX = xs
            .mul(entradasMax.sub(entradasMin))
            .add(entradasMin);
        
        const desnormY = preds
            .mul(etiquetasMax.sub(etiquetasMin))
            .add(etiquetasMin);
        
        return [ desnormX.dataSync(), desnormY.dataSync() ];
    });

    const puntosPrediccion = Array.from(xs).map((val, i) => {
        return { x: val, y: preds[i] }
    });

    const puntosOriginales = data.map(d => ({
        x: d.cuartos, y: d.precio,
    }));

    tfvis.render.scatterplot(
        { name: 'Predicciones vs Originales' },
        { values: [ puntosOriginales, puntosPrediccion ], series: ['originales', 'preddiciones'] },
        {
            xLabel: 'Cuartos',
            yLabel: 'Precio',
            height: 300
        },
    );
}

async function guardarModelo() {
    const saveResult = await modelo.save('downloads://modelo-regresion');
}

async function cargarModelo() {
    const uploadJSONInput = document.getElementById('upload-json');
    const uploadWeightsInput = document.getElementById('upload-weights');
    modelo = await tf.loadLayerModel(tf.io.browserFiles(
        [ uploadJSONInput.files[0], uploadWeightsInput.files[0] ]
    ));

    console.log("Modelo cargado");
}

let modelo;
async function run() {
    const data = await getData();

    visualizarDatos(data);
    modelo = crearModelo();
    const tensorData = convertirDatosATensores(data);
    const { entradas, etiquetas } = tensorData;
    entrenarModelo(modelo, entradas, etiquetas);
}

run();