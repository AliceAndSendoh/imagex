const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
//定义buffer转化为 tensor 的方法
const img2x = (imgPath) =>{
    const buffer = fs.readFileSync(imgPath);
    //tidy优化性能
    return tf.tidy(() =>{
        const imgTs = tf.node.decodeImage(new Uint8Array(buffer));
        //图片大小调整
        const imgTsResized = tf.image.resizeBilinear(imgTs,[224,224]);
        //归一化
        return imgTsResized.toFloat().sub(255/2).div(255/2).reshape([1,224,224,3]);
    });
}
const getData = async (trainDir,outputDir) => {
    const classes =  fs.readdirSync(trainDir).filter(n => !n.includes('.'))
    //读取目录
    fs.writeFileSync(`${outputDir}/classes.json`, JSON.stringify(classes));

    const data = [];
    //分文件夹读取jpg文件
    classes.forEach((dir,dirIndex) => {
        //为节约测试时间，只去每个文件夹前20个文件
       fs.readdirSync(`${trainDir}/${dir}`).slice(0,500)
           .filter(n => n.match(/jpg$/))
           .forEach(filename => {
               const imgPath = `${trainDir}/${dir}/${filename}`;
               data.push({imgPath,dirIndex});//读取图片路径
           });
    });
    tf.util.shuffle(data);//数据洗牌
    //创建dataset
    const ds = tf.data.generator(function* (){
        const count = data.length;
        const batchSize = 32;
        for(let start = 0; start < count; start+=batchSize){
            const end = Math.min(start + batchSize,count);
           yield tf.tidy(() =>{
               const inputs =[];
               const labels =[];
               for(let j = start;j<end;j+=1){
                   const {imgPath,dirIndex}=data[j];
                   const x = img2x(imgPath);
                   inputs.push(x);
                   labels.push(dirIndex);
               }
               const xs = tf.concat(inputs);
               const ys = tf.tensor(labels);
               return {xs,ys};
           });
        }
    })

    return {
        ds,
        classes
    }

};

module.exports = getData;
