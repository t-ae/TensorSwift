import Darwin

public func **(lhs: Tensor, rhs: Tensor) -> Tensor {
    return noncommutativeBinaryOperation(lhs, rhs, operation: powf)
}

public func **(lhs: Tensor, rhs: Tensor.Element) -> Tensor {
    return Tensor(shape: lhs.shape, elements: lhs.elements.map { powf($0, rhs) })
}

public func **(lhs: Tensor.Element, rhs: Tensor) -> Tensor {
    return Tensor(shape: rhs.shape, elements: rhs.elements.map { powf(lhs, $0) })
}

extension TensorProtocol {

    public func sin() -> Tensor {
        return Tensor(shape: shape, elements: self.map(sinf))
    }

    public func cos() -> Tensor {
        return Tensor(shape: shape, elements: self.map(cosf))
    }

    public func tan() -> Tensor {
        return Tensor(shape: shape, elements: self.map(tanf))
    }

    public func asin() -> Tensor {
        return Tensor(shape: shape, elements: self.map(asinf))
    }
    
    public func acos() -> Tensor {
        return Tensor(shape: shape, elements: self.map(acosf))
    }
    
    public func atan() -> Tensor {
        return Tensor(shape: shape, elements: self.map(atanf))
    }
    
    public func sinh() -> Tensor {
        return Tensor(shape: shape, elements: self.map(sinhf))
    }
    
    public func cosh() -> Tensor {
        return Tensor(shape: shape, elements: self.map(coshf))
    }
    
    public func tanh() -> Tensor {
        return Tensor(shape: shape, elements: self.map(tanhf))
    }
    
    public func exp() -> Tensor {
        return Tensor(shape: shape, elements: self.map(expf))
    }
    
    public func log() -> Tensor {
        return Tensor(shape: shape, elements: self.map(logf))
    }
    
    public func sqrt() -> Tensor {
        return Tensor(shape: shape, elements: self.map(sqrtf))
    }
    
    public func cbrt() -> Tensor {
        return Tensor(shape: shape, elements: self.map(cbrtf))
    }
}

extension Tensor {
    public func sigmoid() -> Tensor {
        return Tensor(shape: shape, elements: elements.map { 1.0 / (1.0 + expf(-$0)) })
    }
}
