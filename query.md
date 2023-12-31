| Roberta 解析对照上                     | feature | ground truth        |      |
| -------------------------------------- | ------- | ------------------- | ---- |
| Query: the cars in front of ours       |         | json 文件的名称类型 |      |
| Query: the left cars which are parking |         |                     |      |
| Query: the left cars in black          |         |                     |      |
| Query: the parking cars                |         |                     |      |
| Query: the pedestrian                  |         |                     |      |
|                                        |         |                     |      |
|                                        |         |                     |      |
|                                        |         |                     |      |
|                                        |         |                     |      |

总结，training benchmark 里的

testcase: 3video from 21 MOT in KITTI

Query: the <mask> <moving object> which are <mask>

CLIP: 

input a specific car, 他的属性如何填到<mask> 里

- color : 

the cars in <mask> of ours

- CLIP : tokenized vector, processed by roberta and the whole network 
  - need a loss function to justify it



## Descision tree



