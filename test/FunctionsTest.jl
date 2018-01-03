@testset "Functions" begin
    Base.info("Testing functions...")
    inputvec = [-5.0, 0.0, 5.0]
    @test sigmoid(inputvec[2]) == 0.5
    @test sigmoid.(inputvec) == map(x -> 1.0 / (1.0 + exp(-x)), inputvec)
    @test relu(inputvec[1]) == zero(inputvec[1])
    @test relu.(inputvec) == [0.0, 0.0, inputvec[3]]

    inputvec_softmax = [0.0, 0.5, 1.0]
    @test softmax(inputvec_softmax) == [0.36787944117144233, 0.6065306597126334, 1.0] ./ 1.9744101008840758
    @test onehot([1,3,2], 1:3) == [[1,0,0] [0,0,1] [0,1,0]]
    @test crossentropyerror([1,0,0], [1,0,0]) == -log(1.0 + 1.0e-7)
    @test crossentropyerror([0.6,0.4,0], [0,0,1]) == -log(1.0e-7)
    Base.info("Done testing functions.")
end
