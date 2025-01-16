classdef expLayer < nnet.layer.Layer
    methods
        function layer = expLayer(name)
            % Constructor for the exponential layer
            layer.Name = name;
        end

        function Z = predict(layer, X)
            % Forward pass: Apply the exponential function
            Z = exp(X);
        end

        function dLdX = backward(layer, X, ~, dLdZ, ~)
            % Backward pass: Compute the gradient of the loss w.r.t. the input
            dLdX = dLdZ .* exp(X);
        end
    end
end


