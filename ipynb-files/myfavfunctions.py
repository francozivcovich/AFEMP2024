import numpy as np
import scipy.sparse as sps

def simplexQuadrature( N, D ):
# Created by Greg von Winckel, translated in Python by Franco Zivcovich.
#
# Original code:
# https://www.mathworks.com/matlabcentral/fileexchange/9435-n-dimensional-simplex-quadrature
#
# Translation in Julia:
# https://github.com/eschnett/SimplexQuad.jl
#
        def rquad( N, d ):
            #
            cof = 2 * np.array( range( N ) ) + d
            if d:
                A = ( d ** 2 ) / ( cof * ( cof + 2 ) )
            else:
                A = np.zeros( N )
            B = ( cof[1:] ** 2 - d ** 2 ) / ( 2 * cof[1:] * np.sqrt( cof[1:]**2 - 1 ) )
            M = np.diag( A ) + np.diag( B, -1 ) + np.diag( B, 1 )
            #
            x, V = np.linalg.eig( M )
            i = np.argsort( x )
            x = ( x[ i ] + 1 ) / 2
            w = V[ 0,i ]**2 / ( d + 1 )
            return x, w
        #
        X = np.ones( ( N ** D, D + 1 ) )
        W = np.ones( ( N ** D, ) )
        dim = np.ones( D, dtype = 'int32' )
        dim[0] = -1
        for d in range( 1, D + 1 ):
            # cornerstone of this implementation
            x, w = rquad( N, D - d )
            # some auxiliary arrays
            transposition = np.array( range( D ), dtype = 'int32' )
            transposition[ d - 1 ] = 0
            transposition[ 0 ] = d - 1
            tilation = np.ones( D, dtype = 'int32' ) * N
            tilation[ d - 1 ] = 1
            # create the structures you need
            x = np.transpose( x.reshape( dim ), transposition )
            w = np.transpose( w.reshape( dim ), transposition )
            # build X and W
            X[ :,d ] = X[ :,d-1 ] * np.tile( x, tilation ).reshape( -1, order = 'F' )
            W        = W          * np.tile( w, tilation ).reshape( -1, order = 'F' )

        # manipulate output
        X = - np.diff( X,1,1 )
        permutation = np.roll( np.array( range( D ), dtype = 'int32' ),1 )
        X = X[:,permutation]

        return X, W

def gradphi( d,x ):
    # gradient of the basis function
    return np.hstack( ( - np.ones( ( d,1 ) ), np.eye( d ) ) )

def phi( d,x ):
    # basis function
    return np.eye( d + 1, 1 ).reshape(-1) + x @ gradphi( d,x )

def mass_assembly( Points, CList, dtB ):

    Nv     = Points.shape[0]
    Nc     =  CList.shape[0]
    d_elem =  CList.shape[1] - 1

    i = np.kron( np.ones( ( 1,CList.shape[1] ), dtype = 'int64' ), CList ).reshape(-1,order='F')
    j = np.kron( CList, np.ones( ( 1,CList.shape[1] ), dtype = 'int64' ) ).reshape(-1,order='F')

    X, W = simplexQuadrature( 2, d_elem )
    mass = np.zeros( ( Nc, CList.shape[1], CList.shape[1] ) )
    for q in range( W.shape[0] ):
        N0 = phi( d_elem, X[q,:][None,:] )
        mass = mass + ( N0.T @ N0 ) * ( dtB * W[q] )[:,None,None]
    M = sps.csr_matrix( ( mass.reshape(-1,order='F'), ( i, j ) ), shape = ( Nv, Nv ), dtype = np.float64 )

    return M

def stiff_assembly( Points, CList, dtB, iBt, sigma = None ):

    Nv     = Points.shape[0]
    Nc     =  CList.shape[0]
    d_embe = Points.shape[1]
    d_elem =  CList.shape[1] - 1

    i = np.kron( np.ones( ( 1,CList.shape[1] ), dtype = 'int64' ), CList ).reshape(-1,order='F')
    j = np.kron( CList, np.ones( ( 1,CList.shape[1] ), dtype = 'int64' ) ).reshape(-1,order='F')

    if np.logical_not( bool( sigma ) ):
        sigma = np.stack( Nc * tuple( np.eye(d_embe)[None,:,:] ), axis = 0 )

    X, W = simplexQuadrature( 1, d_elem )
    stiff = np.zeros( ( Nc, CList.shape[1], CList.shape[1] ) )
    for q in range( W.shape[0] ):
        N1 = iBt @ gradphi( d_elem, X[q,:][None,:] )
        stiff = stiff - ( N1.transpose(0,2,1) @ (sigma @ N1) ) * ( dtB * W[q] )[:,None,None]
    S = sps.csr_matrix( ( stiff.reshape(-1,order='F'), ( i, j ) ), shape = ( Nv, Nv ), dtype = np.float64 )

    return S


def external_subsimplexes( CList ):

    d = CList.shape[1] - 1

    # Build subsimplexes List
    if ( d == 2 ):
        SList = np.array( [[1,2],[2,0],[0,1]] )
    if ( d == 3 ):
        SList = np.array([[1,3,2],[2,3,0],[0,3,1],[1,2,0]])

    # let's produce ALL subsimplexes in our mesh
    # (alot of them are repeated twice cause they're busy kissing)
    all_subsimp = CList[:,SList.reshape(-1)].reshape((-1,d))

    # now we sort each edge list so that they are increasing array
    all_subsimp = np.sort( all_subsimp, axis=1 )

    j = np.lexsort( np.flip( all_subsimp, axis = 1 ).T )

    # now we diff all_subsimp[j,:] to find repeated subsimp (if any is nonzero then is not repeated)
    # the 1st one cannot be repeated already, say the 2nd one is equal to the 1st then the 2nd is
    # taken to be the repeated!
    twins = np.hstack( ( True, np.any( np.diff( all_subsimp[ j,: ], axis = 0 ), axis = 1 ) ) )
    outer = np.hstack( ( twins[:-1] * twins[1:], twins[-1] ) )

    inv_j = np.empty_like( j )
    inv_j[ j ] = np.arange( j.size )
    outer = outer[ inv_j ]

    # here we do differently from the MATLAB implementation  precisely for
    # keeping the output the same through the code
    id = np.flip( np.argwhere( outer.reshape( ( d + 1,-1 ), order = 'F' ) ), axis = 1 )


    return all_subsimp[ np.argwhere( outer ).reshape(-1), : ]


def mesh_renumbering( Points, CList ):

        Nv     = Points.shape[0]
        Nc     =  CList.shape[0]
        d_elem =  CList.shape[1] - 1

        i = np.kron( np.ones( ( 1,CList.shape[1] ), dtype = 'int64' ), CList ).reshape(-1,order='F')
        j = np.kron( CList, np.ones( ( 1,CList.shape[1] ), dtype = 'int64' ) ).reshape(-1,order='F')
        id = np.argwhere( i != j ).reshape(-1)
        C = sps.csc_matrix( ( np.ones( id.shape, dtype = bool ), ( i[id], j[id] ) ), dtype = bool ).tocoo()
        i = C.row
        j = C.col
        vertexDegree = np.bincount( i, np.ones( i.shape, dtype = bool ), Nv )

        where_j = np.hstack( ( 0, 1 + np.argwhere( np.diff( j ) ).reshape(-1), j.shape[0] ) ).astype('int64')
        c = np.ones ( Nv, dtype = bool )

        Q = np.zeros( Nv, dtype = 'int64' )
        starting_vertex = np.argmin( vertexDegree )

        count = 0
        Q[ count ] = starting_vertex
        c[ Q[ count ] ] = False

        head = -1
        while ( count < Nv - 1 ):
            head = head + 1
            adj = i[ where_j[ Q[head] ] : where_j[ Q[head] + 1 ] ];
            id = np.argsort( vertexDegree[ adj ], kind='stable' )
            for k in range( adj.shape[0] ):
                if c[ adj[ id[ k ] ] ]:
                    count = count + 1
                    Q[ count ] = adj[ id[k] ]
                    c[ Q[ count ] ] = False

        reverseCMK = True
        if reverseCMK:
            Q = np.flip( Q )

        Points = Points[ Q,: ]

        iQ = np.empty_like( Q )
        iQ[Q] = np.arange( Q.size )
        CList = iQ[ CList.reshape(-1) ].reshape( CList.shape )

        return Points, CList
