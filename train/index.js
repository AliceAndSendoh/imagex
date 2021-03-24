
const tf = require('@tensorflow/tfjs-node');
const getData = require('./loadData');
const TRAIN_DIR = 'waste_sorting/train';
const OUTPUT_DIR = 'output';
const MOBILENET_URL = 'http://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/mobilenet/web_model/model.json';

const main = async () => {
    //load data
    const {ds,classes } = await getData(TRAIN_DIR,OUTPUT_DIR);
    //design model
    const mobilenet = await tf.loadLayersModel(MOBILENET_URL);
    mobilenet.summary();
    console.log(mobilenet.layers.map((l,i) =>[l.name,i]));
    const model = tf.sequential();//定义一个连续模型
    for(let i = 0; i<=86; i+=1){
        const layer = mobilenet.layers[i];
        layer.trainable = false;
        model.add(layer);
    }
    //高维数据摊平
    model.add(tf.layers.flatten());
    //以下是双层神经网络，做分类
    //隐藏层
    model.add(tf.layers.dense({
        units :10,
        activation:'relu',
    }));
    //输出层
    model.add(tf.layers.dense({
        units:classes.length,
        activation:'softmax',
    }));
    //train model
    model.compile({
        loss:'sparseCategoricalCrossentropy',
        optimizer:tf.train.adam(),
        metrics:['acc']
    });
    await model.fitDataset(ds,{epochs:20});
    await model.save(`file://${process.cwd()}/${OUTPUT_DIR}`);

}

main();