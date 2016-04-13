import Darwin
import Accelerate

extension Tensor {
    public var softmax: Tensor {
        let exps = exp
        let sum = exps.elements.reduce(0.0, combine: +)
        return exps / sum
    }
    
    public var relu: Tensor {
        return Tensor(shape: shape, elements: elements.map { fmax($0, 0.0) })
    }
}

extension Tensor {
    public func maxPool(ksize ksize: [Int], strides: [Int]) -> Tensor { // padding = Same
        assert(shape.dimensions.count == 4, "`shape.dimensions.count` must be 4: \(shape.dimensions.count)")
        assert(ksize.count >= 4, "`ksize.count` must be greater than or equal to 4: \(ksize.count)")
        assert(strides.count >= 4, "`strides.count` must be greater than or equal to 4: \(strides.count)")
        assert(strides[0] == 1 ,"`strides[0]` must be 1")
        assert(strides[3] == 1 ,"`strides[3]` must be 1")
        assert(ksize[0] == 1 ,"`ksize[0]` must be 1")
        assert(ksize[3] == 1 ,"`ksize[3]` must be 1")
        
        
        let numBatches = Int(ceil(Float(shape.dimensions[0].value) / Float(strides[0])))
        let numRows = Int(ceil(Float(shape.dimensions[1].value) / Float(strides[1])))
        let numCols = Int(ceil(Float(shape.dimensions[2].value) / Float(strides[2])))
        let numChannels = Int(ceil(Float(shape.dimensions[3].value) / Float(strides[3])))
        
        let padAlongHeight = (numRows - 1) * strides[1] + ksize[1] - shape.dimensions[1].value
        let padAlongWidth = (numCols - 1) * strides[2] + ksize[2] - shape.dimensions[2].value
        let padTop = padAlongHeight / 2
        let padLeft = padAlongWidth / 2
        
        // maximizeするのでmin valueで初期化
        let elements = [Element](count: numBatches * numCols * numRows * numChannels, repeatedValue: FLT_MIN)
        
        for b in 0..<numBatches {
            var elementIndexI = b * numRows
            for i in 0..<numRows {
                for di in 0..<ksize[1] {
                    let y = i+di - padTop
                    if(y<0){
                        continue
                    }
                    if(y>=self.shape.dimensions[1].value){
                        break
                    }
                    var elementIndexJ = elementIndexI * numCols
                    for j in 0..<numCols {
                        var selfIndex = b
                        selfIndex = selfIndex * self.shape.dimensions[1].value + y
                        selfIndex = selfIndex * self.shape.dimensions[2].value + max(0, j-padLeft)
                        selfIndex = selfIndex * self.shape.dimensions[3].value
                        var selfPointer = UnsafeMutablePointer<Element>(self.elements) + selfIndex
                        for dj in 0..<ksize[2] {
                            let x = j+dj - padLeft
                            if(x<0 || x>=self.shape.dimensions[2].value){
                                continue
                            }
                            var elementPointer = UnsafeMutablePointer<Element>(elements) + elementIndexJ * numChannels
                            for _ in 0..<numChannels {
                                elementPointer.memory = max(elementPointer.memory, selfPointer.memory)
                                elementPointer += 1
                                selfPointer += 1
                            }
                        }
                        elementIndexJ += 1
                    }
                }
                elementIndexI += 1
            }
        }
        
        return Tensor(shape: [Dimension(numBatches) ,Dimension(numRows), Dimension(numCols), Dimension(numChannels)], elements: elements)
    }
    
    
    public func conv2d(filter filter: Tensor, strides: [Int]) -> Tensor { // padding = Same
        assert(shape.dimensions.count == 4, "`shape.dimensions.count` must be 4: \(shape.dimensions.count)")
        assert(filter.shape.dimensions.count == 4, "`filter.shape.dimensions.count` must be 4: \(filter.shape.dimensions.count)")
        assert(strides.count >= 4, "`strides.count` must be greater than or equal to 4: \(strides.count)")
        assert(strides[0] == 1 ,"`strides[0]` must be 1")
        assert(strides[3] == 1 ,"`strides[3]` must be 1")
        
        let numBatches = Int(ceil(Float(shape.dimensions[0].value) / Float(strides[0])))
        let numRows = Int(ceil(Float(shape.dimensions[1].value) / Float(strides[1])))
        let numCols = Int(ceil(Float(shape.dimensions[2].value) / Float(strides[2])))
        let numOutChannels = filter.shape.dimensions[3].value
        
        let padAlongHeight = (numRows - 1) * strides[1] + filter.shape.dimensions[0].value - shape.dimensions[1].value
        let padAlongWidth = (numCols - 1) * strides[2] + filter.shape.dimensions[1].value - shape.dimensions[2].value
        let padTop = padAlongHeight / 2
        let padLeft = padAlongWidth / 2
        
        let elements = [Element](count: numBatches * numCols * numRows * numOutChannels, repeatedValue: 0)
        
//      https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#conv2d
//        output[b, i, j, k] =
//            sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
//                filter[di, dj, q, k]
        
        
        for b in 0..<numBatches {
            // インデックス計算を途中までaccumulate
            let selfIndexB = b * shape.dimensions[1].value
            var pointerIndexI = b * numRows
            
            for i in 0..<numRows {
                for di in 0..<filter.shape.dimensions[0].value { // filter height
                    let y = strides[1]*i+di - padTop
                    if(y<0){
                        continue
                    }
                    if(y>=self.shape.dimensions[1].value){
                        // Yが大きい場合それ以降も常に大きいのでbreak
                        break
                    }
                    // インデックス計算を途中までaccumulate
                    let selfIndexY = (selfIndexB + y) * shape.dimensions[2].value
                    let filterIndexDI = di * filter.shape.dimensions[1].value
                    var pointerIndexJ = pointerIndexI * numCols
                    
                    for j in 0..<numCols {
                        // xが確定する前にポインタを作れる(=xもシーケンシャルアクセス)
                        let selfIndex = (selfIndexY + max(0, strides[2]*j - padLeft)) * shape.dimensions[3].value
                        var selfPointer = UnsafeMutablePointer<Element>(self.elements) + selfIndex
                        
                        for dj in 0..<filter.shape.dimensions[1].value { // filter width
                            let x = strides[2]*j+dj - padLeft
                            if(x < 0 || x>=self.shape.dimensions[2].value){
                                continue
                            }
                            // filterのポインタ
                            let filterIndex = (filterIndexDI + dj) * filter.shape.dimensions[2].value * filter.shape.dimensions[3].value
                            var filterPointer = UnsafeMutablePointer<Element>(filter.elements) + filterIndex
                            for _ in 0..<filter.shape.dimensions[2].value { // in channels (loop of q)
                                // elementsのポインタ
                                var pointer = UnsafeMutablePointer<Element>(elements) + pointerIndexJ * numOutChannels
                                for _ in 0..<numOutChannels { // loop of k
                                    pointer.memory += selfPointer.memory * filterPointer.memory
                                    // kの増加でインクリメント
                                    pointer += 1
                                    filterPointer += 1
                                }
                                // qの増加でインクリメント
                                selfPointer += 1
                            }
                        }
                        pointerIndexJ += 1
                    }
                }
                pointerIndexI += 1
            }
        }
        
        return Tensor(shape: [Dimension(numBatches) ,Dimension(numRows), Dimension(numCols), Dimension(numOutChannels)], elements: elements)
    }
    
    public func conv2d_fast(filter filter: Tensor, strides: [Int]) -> Tensor { // padding = Same
        assert(shape.dimensions.count == 4, "`shape.dimensions.count` must be 4: \(shape.dimensions.count)")
        assert(filter.shape.dimensions.count == 4, "`filter.shape.dimensions.count` must be 4: \(filter.shape.dimensions.count)")
        assert(strides.count >= 4, "`strides.count` must be greater than or equal to 4: \(strides.count)")
        assert(strides[0] == 1 ,"`strides[0]` must be 1")
        assert(strides[3] == 1 ,"`strides[3]` must be 1")
        
        let numBatches = Int(ceil(Float(shape.dimensions[0].value) / Float(strides[0])))
        let numRows = Int(ceil(Float(shape.dimensions[1].value) / Float(strides[1])))
        let numCols = Int(ceil(Float(shape.dimensions[2].value) / Float(strides[2])))
        let numOutChannels = filter.shape.dimensions[3].value
        
        let padAlongHeight = (numRows - 1) * strides[1] + filter.shape.dimensions[0].value - shape.dimensions[1].value
        let padAlongWidth = (numCols - 1) * strides[2] + filter.shape.dimensions[1].value - shape.dimensions[2].value
        let padTop = padAlongHeight / 2
        let padLeft = padAlongWidth / 2
        
        // numRows*numCols x フィルタのheight*width*in　の行列として使用
        let _rowSize = filter.shape.dimensions[0].value * filter.shape.dimensions[1].value * filter.shape.dimensions[2].value
        let X = [Float](count: numRows * numCols * _rowSize, repeatedValue: 0)
        var X_pointer = UnsafeMutablePointer<Float>(X)
        for y in 0..<numRows {
            for x in 0..<numCols{
                for c in 0..<filter.shape.dimensions[0].value{ // パッチ画像の何行目か
                    
                    let inputY = y*strides[1] + c - padTop // パディングなし入力画像中のy座標
                    if(inputY < 0 || inputY >= shape.dimensions[1].value){
                        // y方向にはみ出ていたらコピーする必要なし
                        X_pointer += filter.shape.dimensions[1].value * shape.dimensions[3].value
                        continue
                    }
                    
                    var inputX = x*strides[2] - padLeft // パディングなし入力画像中のx座標
                    
                    // パディング付き入力画像上の一行をXの行の対応部分にコピー
                    var startIndex = 0 // コピー先の開始位置相対指定
                    var pixelCount = filter.shape.dimensions[1].value // コピーするピクセル数
                    
                    if(inputX < 0){
                        // 左にはみ出てるxの分スタートとカウントをずらす。
                        startIndex += -inputX * shape.dimensions[3].value
                        pixelCount -= -inputX
                        inputX = 0
                    }
                    if(inputX + pixelCount > shape.dimensions[2].value){
                        // 右にはみ出ている分のカウントを減らす
                        pixelCount -= (inputX + pixelCount - shape.dimensions[2].value)
                    }
                    
                    let imageStartIndex = ((inputY * shape.dimensions[2].value) + inputX) * shape.dimensions[3].value
                    
                    let source = UnsafePointer<Float>(self.elements) + imageStartIndex
                    X_pointer += startIndex
                    memcpy(X_pointer, source, pixelCount * shape.dimensions[3].value * sizeof(Float))
                    X_pointer += filter.shape.dimensions[1].value * shape.dimensions[3].value - startIndex
                }
            }
        }
        
        if(false){
            print(X)
        }
        
        let a = UnsafePointer<Float>(X)
        let b = UnsafePointer<Float>(filter.elements)
        
        let z = Tensor(shape: [Dimension(numBatches), Dimension(numRows), Dimension(numCols), Dimension(numOutChannels)])
        
        let c = UnsafeMutablePointer<Float>(z.elements)
        
        let M = numRows * numCols
        let N = numOutChannels
        let K = _rowSize
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(M), Int32(N), Int32(K), 1.0,
                    a, Int32(K),
                    b, Int32(N), 1.0,
                    c, Int32(N))
        
        return z
    }
}